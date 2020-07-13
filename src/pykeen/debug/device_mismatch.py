from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

import pandas as pd
import torch
import numpy as np

from pathlib import Path

result = pipeline(
    dataset='Kinships',
    model='RotatE',
    random_seed=1235,
    training_kwargs=dict(num_epochs=1)
)

model = result.model

d = model.predict_tails('person10', 'term22')
print(d)