"""Goal, turn ChEBI, ChEMBL, and PubChem IDs into SMILES."""

from collections.abc import Sequence
from functools import lru_cache

import h5py
import pandas as pd
import pyobo
import pystow
import torch
from protmapper import uniprot_client
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from pykeen.models import ERModel
from pykeen.nn import BackfillRepresentation, Embedding, init
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory

HUMAN_T5_PROTEIN_EMBEDDINGS = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/UP000005640_9606/per-protein.h5"
CHEBI_STRUCTURES = "https://ftp.ebi.ac.uk/pub/databases/chebi/Flat_file_tab_delimited/chebiId_inchi.tsv"
EXCAPE_URL = "https://zenodo.org/record/2543724/files/pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv.xz"


def get_human_protein_embedding(uniprot_ids: Sequence[str] | None = None) -> tuple[Embedding, dict[str, int]]:
    """Get an embedding object for human proteins.

    :param uniprot_ids: A sequence of UniProt protein identifiers (like `Q13506`) to get the embeddings for.
        If none are given, will retrieve all human proteins.
    :return: A pair of an embedding object and a mapping from UniProt protein identifier strings to their
        respective positions in the embedding.
    """
    path = pystow.ensure("bio", "uniprot", url=HUMAN_T5_PROTEIN_EMBEDDINGS)
    with h5py.File(path, "r") as file:
        if uniprot_ids is None:
            uniprot_ids = sorted(file)
        tensor = torch.stack([torch.tensor(file[uniprot_id]) for uniprot_id in tqdm(uniprot_ids)])
        uniprot_to_id = {uniprot_id: i for i, uniprot_id in enumerate(uniprot_ids)}
        embedding = Embedding.from_pretrained(tensor)
    return embedding, uniprot_to_id


HEADER = [
    "Ambit_InchiKey",  # hashkey
    "Original_Entry_ID",  # source database id
    "pXC50",  # measurement value (float)
    "DB",  # source database + version
    "InChI",  # Structual information
    "SMILES",  # Same thing
    "Entrez_ID",  # Identifer from the entrez database for the target
    "Tax_ID",  # Species
    "Gene_Symbol",  # pretty name for gene
    "Ortholog_Group",  # Gene group classifier
    "Activity_Flag",  # active / not active
    "Original_Assay_ID",  # Identifier of original assay
]


def get_excape_edges():
    path = pystow.ensure("bio", "excape", url=EXCAPE_URL)

    df = pd.read_csv(
        path,
        sep="\t",
        compression="xz",
        columns=["DB", "Original_Entry_ID", "SMILES", "Entrez_ID", "Activity_Flag"],
        dtype=str,
    )
    # keep only active relationships
    df = df[df["Activity_Flag"] == "A"]
    # slice down to human targets
    df = df[df["Tax_ID"] == "9606"]

    df["uniprot"] = df["Entrez_ID"].map(uniprot_client.get_id_from_entrez)


def main() -> None:
    """Demonstrate using chemical representations for a subset of entities."""
    get_human_protein_embedding(["Q13506", "Q13507", "Q13508", "Q13509"])
    return

    edges_df = pyobo.get_edges_df("chebi")

    tf = TriplesFactory.from_labeled_triples(edges_df.values)

    curie_to_smiles = _get_chebi_smiles()
    # curie_to_smiles.update(_get_slm_smiles())

    example_curie = "chebi:1000"
    tf.entity_labeling.label_to_id[example_curie]

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


@lru_cache(1)
def _get_chebi_smiles() -> dict[str, str]:
    df = pyobo.get_properties_df("chebi")
    df = df[df["property"] == "smiles"]
    return {f"chebi:{k}": v for k, v in df[["chebi_id", "value"]].values}


@lru_cache(1)
def _get_slm_smiles() -> dict[str, str]:
    df = pyobo.get_properties_df("slm")
    df = df[df["property"] == "ChEMROF:smiles_string"]
    return {f"slm:{k}": v for k, v in df[["slm_id", "value"]].values}


def encode_smiles(smiles: str) -> torch.BoolTensor | None:
    """Encode a SMILES string representing a molecule as a boolean tensor."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    vector = MACCSkeys.GenMACCSKeys(molecule)
    return torch.tensor(list(vector), dtype=torch.bool)


if __name__ == "__main__":
    main()
