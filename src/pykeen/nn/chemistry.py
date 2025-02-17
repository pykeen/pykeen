"""Goal, turn ChEBI, ChEMBL, and PubChem IDs into SMILES."""

import gzip
import itertools as itt
from collections.abc import Sequence
from functools import lru_cache
from typing import NamedTuple

import h5py
import numpy as np
import pandas as pd
import pyobo
import pystow
import torch
from protmapper import uniprot_client
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import ConvertToNumpyArray
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from pykeen.models import ERModel
from pykeen.nn import BackfillRepresentation, Embedding, init
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory

HUMAN_T5_PROTEIN_EMBEDDINGS = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/UP000005640_9606/per-protein.h5"
CHEMBL_EMBEDDINGS = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_35.fps.gz"
EXCAPE_URL = "https://zenodo.org/record/2543724/files/pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv.xz"
GO_URL = "https://current.geneontology.org/annotations/goa_human.gaf.gz"


class EmbeddingBackmap(NamedTuple):
    embedding: Embedding
    backmap: dict[str, int]


def get_human_protein_embedding(uniprot_ids: Sequence[str] | None = None) -> EmbeddingBackmap:
    """Get an embedding object for human proteins.

    :param uniprot_ids: A sequence of UniProt protein identifiers (like `Q13506`) to get the embeddings for.
        If none are given, will retrieve all human proteins.
    :return: A pair of an embedding object and a mapping from UniProt protein identifier strings to their
        respective positions in the embedding. The embeddings are 1024 dimensional.
    """
    path = pystow.ensure("bio", "uniprot", url=HUMAN_T5_PROTEIN_EMBEDDINGS)
    with h5py.File(path, "r") as file:
        if uniprot_ids is None:
            uniprot_ids = sorted(file)
        tensor = torch.stack([torch.tensor(file[uniprot_id]) for uniprot_id in tqdm(uniprot_ids)])
        uniprot_to_id = {uniprot_id: i for i, uniprot_id in enumerate(uniprot_ids)}
        embedding = Embedding.from_pretrained(tensor)
    return EmbeddingBackmap(embedding, uniprot_to_id)


def get_chemical_embedding(chembl_ids: Sequence[str] | None = None) -> EmbeddingBackmap:
    path = pystow.ensure("chembl", "35", url=CHEMBL_EMBEDDINGS)

    if chembl_ids:
        chembl_ids = set(chembl_ids)

    with gzip.open(path, mode="rt") as file:
        for _ in range(6):  # throw away headers
            next(file)

        count = itt.count()
        chembl_id_to_idx = {}
        tensors = []
        for line in tqdm(file, unit_scale=True):
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

    tensor = torch.stack(tensors)
    embedding = Embedding.from_pretrained(tensor)
    return EmbeddingBackmap(embedding, chembl_id_to_idx)


def get_go_annotations():
    """Get relationships between proteins and Gene Ontology terms."""
    path = pystow.ensure("bio", "go", url=GO_URL)
    df = pd.read_csv(path, sep="\t")
    df = df[~df["relation"].str.startswith("NOT")]
    df = df[df["DB"] == "UniProtKB"]
    df = df[["DB Object ID", "Relation", "GO ID"]].rename(
        columns={"DB Object ID": "subject", "Relation": "predicate", "GO ID": "object"}
    )
    print(df["predicate"].unique())
    return df


def process_excape():
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
    excape_df = process_excape()
    chembl_ids = excape_df["chembl"].unique()
    uniprot_ids = excape_df["uniprot"].unique()

    excape_df["predicate"] = "modulates"
    excape_df = excape_df[["chembl", "predicate", "uniprot"]].rename(columns={"chembl": "subject", "uniprot": "object"})

    go_df = get_go_annotations()

    triples_df = pd.concat([excape_df, go_df])

    # example chembls ["CHEMBL465070", "CHEMBL517481", "CHEMBL465069"]
    chemical_base_repr, chembl_id_to_idx = get_chemical_embedding(chembl_ids)

    # example uniprots ["Q13506", "Q13507", "Q13508", "Q13509"]
    protein_base_repr, uniprot_id_to_idx = get_human_protein_embedding(uniprot_ids)

    tf = TriplesFactory.from_labeled_triples(triples_df.values)

    id_to_vec: dict[int, torch.BoolTensor] = {}
    with logging_redirect_tqdm():
        for curie, entity_id in tqdm(
            tf.entity_labeling.label_to_id.items(), unit_scale=True, desc="encoding molecules"
        ):
            smiles = curie_to_smiles.get(curie)
            if smiles is None:
                continue
            vec = encode_smiles(smiles)
            if vec is None:
                continue
            id_to_vec[entity_id] = vec

    entity_ids = sorted(id_to_vec)

    tensor = torch.stack([id_to_vec[entity_id] for entity_id in entity_ids])

    #: dimensionality of learned embedding
    dim = 167

    initializer = init.PretrainedInitializer(tensor)
    base_repr = Embedding(max_id=len(id_to_vec), shape=(dim,), initializer=initializer, trainable=False)

    entity_repr = BackfillRepresentation(
        base_ids=entity_ids,
        max_id=tf.num_entities,
        base=base_repr,
    )

    relation_repr = Embedding(max_id=tf.num_relations, shape=(dim,))

    model = ERModel(
        triples_factory=tf,
        interaction="DistMult",
        entity_representations=entity_repr,
        relation_representations=relation_repr,
    )

    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=tf,
    )
    training_loop.train(tf, num_epochs=5)


if __name__ == "__main__":
    main()
