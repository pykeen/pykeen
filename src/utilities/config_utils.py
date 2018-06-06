import os


def create_base_dir():
    base_dir = os.environ.get('KG_EMBEDDINGS_PIPELINE_DIRECTORY',
                              os.path.join(os.path.expanduser('~'), '.kg_embeddings_pipeline'))
    os.makedirs(base_dir, exist_ok=True)
    return base_dir
