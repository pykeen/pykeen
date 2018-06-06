# Basic Directory
from utilities.config_utils import create_base_dir

KG_EMBEDDINGS_PIPELINE_DIR = create_base_dir()

# Related to corpus Walking RDF and OWL
WROC = 'walking_rdf_and_owl_corpus'
WROC_URL = 'http://aber-owl.net/aber-owl/bio2vec/bio-knowledge-graph.n3'

# Configuration related
CLASS_NAME = 'class_name'
## Reader
READER = 'reader'
WROC_READER = 'WROCReader'
## KG embedding model
KG_EMBEDDING_MODEL = 'kg_embedding_model'
TRANS_E = 'TransE'
TRANS_H = 'TransH'
## ML params
BATCH_SIZE = 'batch_size'
VOCAB_SIZE = 'vocab_size'
EMBEDDING_DIM = 'embedding_dim'
MARGIN_LOSS = 'margin_loss'
NUM_ENTITIES = 'num_entities'
NUM_RELATIONS = 'num_relations'
NUM_EPOCHS = 'num_epochs'
LEARNING_RATE = 'learning_rate'
