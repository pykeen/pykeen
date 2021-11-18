# try:
#     from pykeen.datasets import get_dataset
#     from pykeen.triples.leakage import unleak
#     fb15k = get_dataset(dataset='fb15k')
#     n = 401  # magic 401 from the paper
#     train, test, validate = unleak(fb15k.training, fb15k.testing, fb15k.validation, n=n)
#     quit(0 if train.num_relations == 237 else 1)
# except Exception:
#     quit(125)  # special code for "code cannot be tested"
import logging

from pykeen.datasets import get_dataset
from pykeen.triples.leakage import unleak

logging.basicConfig(format="%(levelname)s %(message)s", level=logging.DEBUG)

fb15k = get_dataset(dataset="fb15k")
n = 401  # magic 401 from the paper
train, test, validate = unleak(fb15k.training, fb15k.testing, fb15k.validation, n=n)
print(train.num_relations, 237)
