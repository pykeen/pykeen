"""This script gives a demo of training a proteochemometrics model with PyKEEN, enriched with GO annotations.

Run with ``uv run --script proteochemometrics.py``.
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "h5py",
#     "numpy",
#     "pandas",
#     "protmapper",
#     "pykeen",
#     "pystow",
#     "rdkit",
#     "tqdm",
# ]
#
# [tool.uv.sources]
# pykeen = { path = "..", editable = true }
# ///

import gzip
import itertools as itt
from collections.abc import Sequence
from typing import NamedTuple

import click
import h5py
import numpy as np
import pandas as pd
import pystow
import torch
from protmapper import uniprot_client
from rdkit import DataStructs
from rdkit.DataStructs import ConvertToNumpyArray
from tqdm import tqdm

from pykeen.models import ERModel
from pykeen.nn import (
    BackfillSpec,
    Embedding,
    FeatureEnrichedEmbedding,
    MultiBackfillRepresentation,
    Representation,
    TransformedRepresentation,
)
from pykeen.predict import predict_target
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory

HUMAN_T5_PROTEIN_EMBEDDINGS = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/UP000005640_9606/per-protein.h5"
CHEMBL_VERSION = "35"
CHEMBL_EMBEDDINGS = f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_{CHEMBL_VERSION}/chembl_{CHEMBL_VERSION}.fps.gz"
EXCAPE_URL = "https://zenodo.org/record/2543724/files/pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv.xz"
GO_URL = "https://current.geneontology.org/annotations/goa_human.gaf.gz"


class RepresentationBackmap(NamedTuple):
    """A representation and a reverse local lookup."""

    embedding: Representation
    labels: list[str]
    label_to_id: dict[str, int]


class MLPTransformedEmbedding(TransformedRepresentation):
    """A representation that transforms an embedding with a simple MLP."""

    def __init__(
        self,
        base: Embedding,
        output_dim: int,
        *,
        ratio: int | float = 2,
    ) -> None:
        """Initialize the representation."""
        hidden_dim = int(ratio * output_dim)
        transformation = torch.nn.Sequential(
            torch.nn.Linear(base.shape[0], hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        super().__init__(
            base=base,
            max_id=base.max_id,
            shape=(output_dim,),
            transformation=transformation,
        )


def get_human_protein_embedding(uniprot_ids: Sequence[str] | None = None) -> RepresentationBackmap:
    """Get an embedding object for human proteins.

    :param uniprot_ids: A sequence of UniProt protein identifiers (like `Q13506`) to get the embeddings for. If none are
        given, will retrieve all human proteins.

    :returns: A pair of an embedding object and a mapping from UniProt protein identifier strings to their respective
        positions in the embedding. The embeddings are 1024 dimensional.
    """
    path = pystow.ensure("bio", "uniprot", url=HUMAN_T5_PROTEIN_EMBEDDINGS)
    with h5py.File(path, "r") as file:
        if uniprot_ids is None:
            uniprot_ids = sorted(file)
        tensor = torch.stack(
            [
                torch.tensor(file[uniprot_id])
                for uniprot_id in tqdm(uniprot_ids, unit_scale=True, unit="protein", desc="Getting protein features")
            ]
        )
        uniprot_id_to_idx = {uniprot_id: idx for idx, uniprot_id in enumerate(uniprot_ids)}
        representation = FeatureEnrichedEmbedding(tensor)
    return RepresentationBackmap(representation, uniprot_ids, uniprot_id_to_idx)


def get_chemical_embedding(chembl_ids: Sequence[str] | None = None) -> RepresentationBackmap:
    """Get an embedding object for chemicals.

    :param chembl_ids: A sequence of ChEMBL chemical identifiers (like `CHEMBL465070`) to get the embeddings for. If
        none are given, will retrieve all ChEMBL chemicals (around 2.5 million)

    :returns: A pair of an embedding object and a mapping from ChEMBL chemical identifier strings to their respective
        positions in the embedding. The embeddings are 2048 dimensional bit vectors from Morgan fingerprints with a
        radius of 2
    """
    path = pystow.ensure("chembl", CHEMBL_VERSION, url=CHEMBL_EMBEDDINGS)
    if chembl_ids is not None:
        chembl_ids = set(chembl_ids)
    with gzip.open(path, mode="rt") as file:
        for _ in range(6):  # throw away headers
            next(file)

        count = itt.count()
        chembl_id_to_idx = {}
        # keep track of this in case of duplicates, missings, or errors
        actual_chembl_ids = []
        tensors = []
        for line in tqdm(file, unit_scale=True, total=2_470_000, desc="Getting chemical features", unit="chemical"):
            hex_fp, chembl_id = line.strip().split("\t")

            if chembl_ids and chembl_id not in chembl_ids:
                continue

            if chembl_id in chembl_id_to_idx:
                tqdm.write(f"duplicate of {chembl_id}")
                continue

            # Convert hex to binary
            binary_fp = bytes.fromhex(hex_fp)

            # Convert binary to RDKit ExplicitBitVect
            bitvect = DataStructs.cDataStructs.CreateFromBinaryText(binary_fp)

            # Convert to NumPy array
            arr = np.zeros((bitvect.GetNumBits(),), dtype=np.uint8)
            ConvertToNumpyArray(bitvect, arr)

            chembl_id_to_idx[chembl_id] = next(count)
            tensors.append(torch.tensor(arr))
            actual_chembl_ids.append(chembl_id)

    tensor = torch.stack(tensors)
    representation = FeatureEnrichedEmbedding(tensor)

    return RepresentationBackmap(representation, actual_chembl_ids, chembl_id_to_idx)


def get_protein_go_triples():
    """Get triples with proteins as the subject and Gene Ontology terms as the objects."""
    path = pystow.ensure("bio", "go", url=GO_URL)
    df = pd.read_csv(
        path, sep="\t", comment="!", usecols=[0, 1, 3, 4], names=["DB", "DB Object ID", "Relation", "GO ID"]
    )
    df = df[~df["Relation"].str.startswith("NOT")]
    df = df[df["DB"] == "UniProtKB"]
    df = df[["DB Object ID", "Relation", "GO ID"]].rename(
        columns={"DB Object ID": "subject", "Relation": "predicate", "GO ID": "object"}
    )
    return df


def get_chemical_protein_triples():
    """Get triples from ExCAPE-DB with chemicals as subjects and proteins as objects."""
    path = pystow.ensure("bio", "excape", url=EXCAPE_URL)
    ff_path = pystow.join("bio", "excape", name="excape_human_chembl_subset.tsv")
    if ff_path.is_file():
        df = pd.read_csv(ff_path, sep="\t")
        return df

    df = pd.read_csv(
        path,
        sep="\t",
        compression="xz",
        usecols=["DB", "Original_Entry_ID", "Entrez_ID", "Activity_Flag", "Tax_ID"],
        dtype=str,
    )

    # keep only active relationships
    df = df[df["Activity_Flag"] == "A"]
    # slice down to human targets
    df = df[df["Tax_ID"] == "9606"]
    # keep only relationships qualified by ChEMBL, even though this way out of date in 2025
    df = df[df["DB"] == "chembl20"]

    df["uniprot"] = df["Entrez_ID"].map(uniprot_client.get_id_from_entrez)

    # throw away anything that can't be mapped to human uniprot
    df = df[df["uniprot"].notna()]

    df = df[["uniprot", "Original_Entry_ID"]].drop_duplicates().rename(columns={"Original_Entry_ID": "chembl"})
    df.to_csv(ff_path, sep="\t", index=False)
    return df


def main() -> None:
    """Demonstrate using chemical representations for a subset of entities."""
    target_dim = 32

    example_chembl_id = "CHEMBL1097808"

    click.echo("Getting chemical-protein triples from ExCAPE-DB")
    excape_df = get_chemical_protein_triples()
    chembl_ids = excape_df["chembl"].unique()
    uniprot_ids = excape_df["uniprot"].unique()

    if example_chembl_id not in chembl_ids:
        raise ValueError("need to pick a chembl ID for prediction that's in ExCAPE-DB")

    click.echo("Getting protein representations from UniProt")
    # example uniprots ["Q13506", "Q13507", "Q13508", "Q13509"]
    protein_base_repr, _, uniprot_ids = get_human_protein_embedding(uniprot_ids)
    protein_trans_repr = MLPTransformedEmbedding(protein_base_repr, target_dim)

    click.echo("Getting chemical representations from ChEMBL")
    # example chembls ["CHEMBL465070", "CHEMBL517481", "CHEMBL465069"]
    chemical_base_repr, _, chembl_ids = get_chemical_embedding(chembl_ids)
    chemical_trans_repr = MLPTransformedEmbedding(chemical_base_repr, target_dim)

    click.echo("Getting protein-GO triples from the Gene Ontology")
    go_df = get_protein_go_triples()

    click.echo("Constructing a triples factory")
    # do some cleanup on the excape df
    excape_df["predicate"] = "modulates"
    excape_df = excape_df[["chembl", "predicate", "uniprot"]].rename(columns={"chembl": "subject", "uniprot": "object"})

    triples_df = pd.concat([excape_df, go_df])
    tf = TriplesFactory.from_labeled_triples(triples_df.values)

    # note that the order you add to this set is very important
    chemical_ids = [
        tf.entity_labeling.label_to_id[chembl_id]
        for chembl_id in chembl_ids
        if chembl_id in tf.entity_labeling.label_to_id
    ]
    click.echo(f"Mapped {len(chemical_ids):,} ChEMBL IDs from edges to Morgan features")

    protein_ids = [
        tf.entity_labeling.label_to_id[uniprot_id]
        for uniprot_id in uniprot_ids
        if uniprot_id in tf.entity_labeling.label_to_id
    ]
    click.echo(f"Mapped {len(protein_ids):,} UniProt IDs from edges to T5 protein features")

    click.echo("Constructing entity representation")
    entity_repr = MultiBackfillRepresentation(
        max_id=tf.num_entities,
        specs=[
            BackfillSpec(chemical_ids, chemical_trans_repr),
            BackfillSpec(protein_ids, protein_trans_repr),
        ],
        backfill_kwargs={"shape": (target_dim,)},
    )

    relation_repr = Embedding(max_id=tf.num_relations, shape=(target_dim,))

    model = ERModel(
        triples_factory=tf,
        interaction="DistMult",
        entity_representations=entity_repr,
        relation_representations=relation_repr,
    )

    click.echo("Constructing training loop")
    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=tf,
    )

    click.echo("Training")
    training_loop.train(tf, num_epochs=1, batch_size=512)

    click.echo("Make some predictions")
    predictions_pack = predict_target(
        model=model,
        head=example_chembl_id,
        relation="modulates",
        triples_factory=tf,
    ).add_membership_columns(training=tf)
    predictions_df = predictions_pack.df
    # TODO use proper CURIEs throughout so filtering is more direct on str.startswith("uniprot:")
    predictions_df = predictions_df[~predictions_df["tail_label"].str.startswith("GO:")]
    click.echo(predictions_df.head(30).to_markdown(index=False))


if __name__ == "__main__":
    main()
