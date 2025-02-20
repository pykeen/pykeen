"""This script gives a demo of training a proteochemometrics model with PyKEEN, enriched with GO annotations.

Run with ``uv run --script main.py``.
"""

# /// script
# requires-python = ">=3.10"
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
# pykeen = { path = "../..", editable = true }
# ///

import gzip
import itertools as itt
from collections.abc import Sequence
from pathlib import Path
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
    Embedding,
    FeatureEnrichedEmbedding,
    MLPTransformedRepresentation,
    MultiBackfillRepresentation,
    Partition,
    Representation,
)
from pykeen.predict import predict_target
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory

HUMAN_T5_PROTEIN_EMBEDDINGS = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/UP000005640_9606/per-protein.h5"
CHEMBL_VERSION = "35"
CHEMBL_EMBEDDINGS = f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_{CHEMBL_VERSION}/chembl_{CHEMBL_VERSION}.fps.gz"
EXCAPE_URL = "https://zenodo.org/record/2543724/files/pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv.xz"
GO_URL = "https://current.geneontology.org/annotations/goa_human.gaf.gz"
HERE = Path(__file__).parent.resolve()


class RepresentationBackmap(NamedTuple):
    """A representation and a reverse local lookup."""

    embedding: Representation
    labels: list[str]
    label_to_id: dict[str, int]


def get_human_protein_embedding(
    uniprot_curies: Sequence[str] | None = None, *, trainable: bool = False
) -> RepresentationBackmap:
    """Get an embedding object for human proteins.

    :param uniprot_curies: A sequence of UniProt protein identifiers (like `Q13506`) to get the embeddings for.
        If none are given, will retrieve all human proteins.
    :param trainable: Should trainable embeddings be combined with the features?

    :returns: A pair of an embedding object and a mapping from UniProt protein identifier strings to their respective
        positions in the embedding. The embeddings are 1024 dimensional.
    """
    path = pystow.ensure("bio", "uniprot", url=HUMAN_T5_PROTEIN_EMBEDDINGS)
    with h5py.File(path, "r") as file:
        if uniprot_curies is None:
            uniprot_curies = sorted(file)
        tensor = torch.stack(
            [
                torch.tensor(file[uniprot_curie.removeprefix("uniprot:")])
                for uniprot_curie in tqdm(
                    uniprot_curies, unit_scale=True, unit="protein", desc="Getting protein features"
                )
            ]
        )
        uniprot_id_to_idx = {uniprot_id: idx for idx, uniprot_id in enumerate(uniprot_curies)}
        if trainable:
            representation = FeatureEnrichedEmbedding(tensor, shape=32)
        else:
            representation = Embedding.from_pretrained(tensor)
    return RepresentationBackmap(representation, [f"uniprot:{i}" for i in uniprot_curies], uniprot_id_to_idx)


def get_chemical_embedding(
    chembl_curies: Sequence[str] | None = None, *, trainable: bool = False
) -> RepresentationBackmap:
    """Get an embedding object for chemicals.

    :param chembl_curies: A sequence of ChEMBL chemical identifiers (like `CHEMBL465070`) to get the embeddings for. If
        none are given, will retrieve all ChEMBL chemicals (around 2.5 million)
    :param trainable: Should trainable embeddings be combined with the features?

    :returns: A pair of an embedding object and a mapping from ChEMBL chemical identifier strings to their respective
        positions in the embedding. The embeddings are 2048 dimensional bit vectors from Morgan fingerprints with a
        radius of 2
    """
    path = pystow.ensure("chembl", CHEMBL_VERSION, url=CHEMBL_EMBEDDINGS)
    if chembl_curies is not None:
        chembl_curies = set(chembl_curies)
    with gzip.open(path, mode="rt") as file:
        for _ in range(6):  # throw away headers
            next(file)

        count = itt.count()
        chembl_curie_to_idx = {}
        # keep track of this in case of duplicates, missings, or errors
        actual_chembl_curies = []
        tensors = []
        for line in tqdm(file, unit_scale=True, total=2_470_000, desc="Getting chemical features", unit="chemical"):
            hex_fp, chembl_id = line.strip().split("\t")
            chembl_curie = f"chembl.compound:{chembl_id}"
            if chembl_curies and chembl_curie not in chembl_curies:
                continue

            if chembl_curie in chembl_curie_to_idx:
                tqdm.write(f"duplicate of {chembl_curie}")
                continue

            chembl_curie_to_idx[chembl_curie] = next(count)
            tensors.append(_hex_to_arr(hex_fp))
            actual_chembl_curies.append(chembl_curie)

    tensor = torch.stack(tensors)
    if trainable:
        representation = FeatureEnrichedEmbedding(tensor, shape=32)
    else:
        representation = Embedding.from_pretrained(tensor)
    return RepresentationBackmap(representation, actual_chembl_curies, chembl_curie_to_idx)


def _hex_to_arr(hex_fp: str) -> torch.BoolTensor:
    # Convert hex to binary
    binary_fp = bytes.fromhex(hex_fp)

    # Convert binary to RDKit ExplicitBitVect
    bitvect = DataStructs.cDataStructs.CreateFromBinaryText(binary_fp)

    # Convert to NumPy array
    arr = np.zeros((bitvect.GetNumBits(),), dtype=np.uint8)
    ConvertToNumpyArray(bitvect, arr)
    return torch.tensor(arr, dtype=torch.bool)


def get_protein_go_triples():
    """Get triples with proteins as the subject and Gene Ontology terms as the objects."""
    path = pystow.ensure("bio", "go", url=GO_URL)
    df = pd.read_csv(
        path, sep="\t", comment="!", usecols=[0, 1, 3, 4], names=["DB", "DB Object ID", "Relation", "GO ID"]
    )
    df = df[~df["Relation"].str.startswith("NOT")]
    df = df[df["DB"] == "UniProtKB"]
    df["subject"] = df["DB Object ID"].map(lambda s: f"uniprot:{s}")
    df = df[["subject", "Relation", "GO ID"]].rename(columns={"Relation": "predicate", "GO ID": "object"})
    return df


def get_chemical_protein_triples():
    """Get triples from ExCAPE-DB with chemicals as subjects and proteins as objects."""
    excape_module = pystow.module("bio", "excape")
    if (path := excape_module.join(name="excape_human_chembl_subset.tsv")).is_file():
        df = pd.read_csv(path, sep="\t")
        return df

    df = excape_module.ensure_csv(
        url=EXCAPE_URL,
        read_csv_kwargs=dict(
            sep="\t",
            compression="xz",
            usecols=["DB", "Original_Entry_ID", "Entrez_ID", "Activity_Flag", "Tax_ID"],
            dtype=str,
        ),
    )
    # keep only active relationships
    df = df[df["Activity_Flag"] == "A"]
    # slice down to human targets
    df = df[df["Tax_ID"] == "9606"]
    # keep only relationships qualified by ChEMBL, even though this way out of date in 2025
    df = df[df["DB"] == "chembl20"]

    df["object"] = df["Entrez_ID"].map(_uniprot_curie_from_entrez)

    # throw away anything that can't be mapped to human uniprot
    df = df[df["object"].notna()]

    # turn into CURIEs
    df["subject"] = df["Original_Entry_ID"].map("chembl.compound:{0}".format)

    df = df[["subject", "object"]].drop_duplicates()
    df.to_csv(path, sep="\t", index=False)
    return df


def _uniprot_curie_from_entrez(s: str | None) -> str | None:
    if pd.isna(s):
        return None
    uniprot_id = uniprot_client.get_id_from_entrez(s)
    return uniprot_id and f"uniprot:{uniprot_id}"


def main() -> None:
    """Demonstrate using chemical representations for a subset of entities."""
    target_dim = 32

    device = torch.device("cpu")

    # set this to true to enrich the chemical and protein features
    # with additional learnable embeddings. Note that this makes
    # training take a _lot_ longer
    enrich_features_with_embedding = True

    example_chembl_id = "chembl.compound:CHEMBL1097808"

    click.echo("Getting chemical-protein triples from ExCAPE-DB")
    excape_df = get_chemical_protein_triples()
    chemical_curies = excape_df["subject"].unique()
    protein_curies = excape_df["object"].unique()

    if example_chembl_id not in chemical_curies:
        raise KeyError(f"{example_chembl_id} is not in ExCAPE-DB. Try {list(chemical_curies[0])}")

    click.echo("Getting protein representations from UniProt")
    # example uniprots ["uniprot:Q13506", "uniprot:Q13507", "uniprot:Q13508", "uniprot:Q13509"]
    protein_base_repr, _, protein_curies = get_human_protein_embedding(
        protein_curies, trainable=enrich_features_with_embedding
    )
    protein_trans_repr = MLPTransformedRepresentation(base=protein_base_repr, output_dim=target_dim)

    click.echo("Getting chemical representations from ChEMBL")
    # example chembls ["chembl.compound:CHEMBL465070", "chembl.compound:CHEMBL517481", "chembl.compound:CHEMBL465069"]
    chemical_base_repr, _, chemical_curies = get_chemical_embedding(
        chemical_curies, trainable=enrich_features_with_embedding
    )
    chemical_trans_repr = MLPTransformedRepresentation(base=chemical_base_repr, output_dim=target_dim)

    click.echo("Getting protein-GO triples from the Gene Ontology")
    go_df = get_protein_go_triples()

    click.echo("Constructing a triples factory")
    # do some cleanup on the excape df
    excape_df["predicate"] = "modulates"
    excape_df = excape_df[["subject", "predicate", "object"]]

    triples_df = pd.concat([excape_df, go_df])
    tf = TriplesFactory.from_labeled_triples(triples_df.values)

    # note that the order you add to this set is very important
    chemical_idx = [
        tf.entity_labeling.label_to_id[chembl_curie]
        for chembl_curie in chemical_curies
        if chembl_curie in tf.entity_labeling.label_to_id
    ]
    click.echo(f"Mapped {len(chemical_idx):,} ChEMBL IDs from edges to Morgan features")

    protein_idx = [
        tf.entity_labeling.label_to_id[uniprot_curie]
        for uniprot_curie in protein_curies
        if uniprot_curie in tf.entity_labeling.label_to_id
    ]
    click.echo(f"Mapped {len(protein_idx):,} UniProt IDs from edges to T5 protein features")

    click.echo("Constructing entity representation")
    entity_repr = MultiBackfillRepresentation(
        max_id=tf.num_entities,
        partitions=[
            Partition(chemical_idx, chemical_trans_repr),
            Partition(protein_idx, protein_trans_repr),
        ],
    ).to(device)

    relation_repr = Embedding(max_id=tf.num_relations, shape=target_dim).to(device)

    model = ERModel(
        triples_factory=tf,
        interaction="DistMult",
        entity_representations=entity_repr,
        relation_representations=relation_repr,
    ).to(device)

    click.echo("Constructing training loop")
    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=tf,
    )

    click.echo("Training")
    losses = training_loop.train(tf, num_epochs=300, batch_size=500_000)
    HERE.joinpath("losses.txt").write_text("\n".join(map(str, losses)))

    model.save_state(HERE.joinpath("model.pkl"))

    click.echo(f"Make some predictions for {example_chembl_id}")
    predictions_pack = predict_target(
        model=model,
        head=example_chembl_id,
        relation="modulates",
        triples_factory=tf,
        targets=protein_idx,
    ).add_membership_columns(training=tf)
    predictions_df = predictions_pack.df
    predictions_df = predictions_df[predictions_df["tail_label"].str.startswith("uniprot:")]
    predictions_df["tail_name"] = predictions_df["tail_label"].map(
        lambda s: uniprot_client.get_gene_name(s.removeprefix("uniprot:"))
    )
    click.echo(predictions_df.head(30).to_markdown(index=False))


if __name__ == "__main__":
    main()
