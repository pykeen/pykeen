# -*- coding: utf-8 -*-
import os
import getpass

def create_base_dir():
    try:
        base_dir = os.environ.get('KG_EMBEDDINGS_PIPELINE_DIRECTORY',
                                  os.path.join('/data', getpass.getuser(), '/project_data/kg_embeddings_pipeline'))
        os.makedirs(base_dir, exist_ok=True)
    except:
        base_dir = os.environ.get('KG_EMBEDDINGS_PIPELINE_DIRECTORY',
                                  os.path.join(os.path.expanduser('~'), '.kg_embeddings_pipeline'))
        os.makedirs(base_dir, exist_ok=True)


    return base_dir
