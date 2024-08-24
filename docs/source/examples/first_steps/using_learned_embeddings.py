"""Using learned embeddings."""

# %%
from pykeen.models import ERModel
from pykeen.pipeline import pipeline

# train a model
result = pipeline(model="TransE", dataset="nations")
model = result.model
assert isinstance(model, ERModel)

# access entity and relation representations
entity_representation_modules = model.entity_representations
relation_representation_modules = model.relation_representations

# %%
from pykeen.nn.representation import Embedding  # noqa: E402

# TransE has one representation for entities and one for relations
# both are simple embedding matrices
entity_embeddings = entity_representation_modules[0]
relation_embeddings = relation_representation_modules[0]
assert isinstance(entity_embeddings, Embedding)
assert isinstance(relation_embeddings, Embedding)

# %%
# get representations for all entities/relations
entity_embedding_tensor = entity_embeddings()
relation_embedding_tensor = relation_embeddings()

# %%
# this corresponds to explicitly passing indices=None
entity_embedding_tensor = entity_embeddings(indices=None)
relation_embedding_tensor = relation_embeddings(indices=None)

# %%
import torch  # noqa: E402

entity_embedding_tensor = entity_embeddings(indices=torch.as_tensor([1, 3]))

# %%
# detach tensor, move to cpu, and convert to numpy
entity_embedding_tensor = entity_embeddings.detach().cpu().numpy()
