import numpy as np
from pykeen.datasets import Dataset
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# Bring Your Own Triples
# -------------------
# It's unlikely that the data you want to work with is already in the PyKEEN package.
# YOu most likely want to train a model on triples you have already prepared.
# This example demonstrates the most basic way to bring your own data to PyKEEN.
# If you have triples that already have relation & entity mappings, you can use the TriplesFactory class constructor directly - that is not shown here.
# For a complete deep dive into Bringing Your Own Data to PyKeen, see https://pykeen.readthedocs.io/en/stable/byo/data.html

# define a set of triples. In this case, we'll make up some data about the Beatles
# The triples are in the form (head, relation, tail)
# The triples should be a numpy array with shape (n, 3)
beatles_triples = np.array([
    ['Paul McCartney', 'member of', 'The Beatles'],
    ['John Lennon', 'member of', 'The Beatles'],
    ['George Harrison', 'member of', 'The Beatles'],
    ['Ringo Starr', 'member of', 'The Beatles'],
    ['The Beatles', 'genre', 'rock'],
    ['The Beatles', 'genre', 'pop'],
    ['The Beatles', 'genre', 'psychedelic rock'],
    ['The Beatles', 'genre', 'art rock'],
    ['The Ed Sullivan Show', 'hosted', 'The Beatles'],
    ['The Beatles', 'formed', 'Liverpool'],
    ['Paul McCartney', 'born in', 'Liverpool'],
    ['John Lennon', 'born in', 'Liverpool'],
    ['George Harrison', 'born in', 'Liverpool'],
    ['Ringo Starr', 'born in', 'Liverpool'],
])

# instantiate a triples factory, using the from_labeled_triples class method
beatles_triples_factory = TriplesFactory.from_labeled_triples(triples=beatles_triples)

# after creating the triples factory, you can examine the triples

# print(beatles_triples_factory.triples)

# Just having the triples might be enough for your use case, if you're training a model directly.
# However, the Dataset abstraction provides an easy way to work with triples in the pipeline.
# instantiate a dataset
beatles_dataset = Dataset.from_tf(beatles_triples_factory)

# Now you can use your own triples in a PyKEEN pipeline.
# For example, you could train a model using the TransE model in a pipeline.
result = pipeline(
    model='TransE',
    dataset=beatles_dataset,
)
