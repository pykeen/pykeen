"""Training a Model with the Pipeline."""

# %%
# The following example shows how to train and evaluate the :class:`~pykeen.models.TransE` model on the
# :class:`~pykeen.datasets.Nations` dataset. Throughout the documentation, you'll notice that each asset has a
# corresponding class in PyKEEN. You can follow the links to learn more about each and see the reference on how to
# use them specifically. Don't worry, in this part of the tutorial, the :func:`~pykeen.pipeline.pipeline` function
# will take care of everything for you.
from pykeen.pipeline import pipeline

pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
)
pipeline_result.save_to_directory("nations_transe")

# %%
# The results are returned in a :class:`~pykeen.pipeline.PipelineResult` instance, which has attributes for the
# trained model, the training loop, and the evaluation.
#
# In this example, the model was given as a string. A list of available models can be found in :mod:`pykeen.models`.
# Alternatively, the class corresponding to the implementation of the model could be used as in:
from pykeen.models import TransE  # noqa: E402

pipeline_result = pipeline(
    dataset="Nations",
    model=TransE,
)
pipeline_result.save_to_directory("nations_transe")

# %%
# In this example, the dataset was given as a string. A list of available datasets can be found in
# :mod:`pykeen.datasets`. Alternatively, a subclass of :class:`~pykeen.datasets.Dataset` could be used as in:
from pykeen.datasets import Nations  # noqa: E402

pipeline_result = pipeline(
    dataset=Nations,
    model=TransE,
)
pipeline_result.save_to_directory("nations_transe")

# %%
# In each of the previous three examples, the training approach, optimizer, and evaluation scheme were omitted. By
# default, the model is trained under the stochastic local closed world assumption (sLCWA;
# :class:`~pykeen.training.SLCWATrainingLoop`). This can be explicitly given as a string:
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    training_loop="sLCWA",
)
pipeline_result.save_to_directory("nations_transe")

# %%
# Alternatively, the model can be trained under the local closed world assumption (LCWA;
# :class:`~pykeen.training.LCWATrainingLoop`) by giving ``'LCWA'``. No additional configuration is necessary, but
# it's worth reading up on the differences between these training approaches. A list of available training
# assumptions can be found in :mod:`pykeen.training`.
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    training_loop="LCWA",
)
pipeline_result.save_to_directory("nations_transe")

# %%
# One of these differences is that the sLCWA relies on *negative sampling*. The type of negative sampling can be
# given as in:
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    training_loop="sLCWA",
    negative_sampler="basic",
)
pipeline_result.save_to_directory("nations_transe")

# %%
# In this example, the negative sampler was given as a string. A list of available negative samplers can be found
# in :mod:`pykeen.sampling`. Alternatively, the class corresponding to the implementation of the negative sampler
# could be used as in:
#
# .. warning ::
#
#    The ``negative_sampler`` keyword argument should not be used if the LCWA is being used. In general, all other
#    options are available under either training approach.
from pykeen.sampling import BasicNegativeSampler  # noqa: E402

pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    training_loop="sLCWA",
    negative_sampler=BasicNegativeSampler,
)
pipeline_result.save_to_directory("nations_transe")

# %%
# The type of evaluation performed can be specified with the ``evaluator`` keyword. By default, rank-based
# evaluation is used. It can be given explicitly as in:
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    evaluator="RankBasedEvaluator",
)
pipeline_result.save_to_directory("nations_transe")

# %%
# In this example, the evaluator was given as a string. A list of available evaluators can be found in
# :mod:`pykeen.evaluation`. Alternatively, the class corresponding to the implementation of the evaluator could be
# used as in:
from pykeen.evaluation import RankBasedEvaluator  # noqa: E402

pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    evaluator=RankBasedEvaluator,
)
pipeline_result.save_to_directory("nations_transe")

# %%
# PyKEEN implements early stopping, which can be turned on with the ``stopper`` keyword argument as in:
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    stopper="early",
)
pipeline_result.save_to_directory("nations_transe")

# %%
# In PyKEEN you can also use the learning rate schedulers provided by PyTorch, which can be turned on with the
# ``lr_scheduler`` keyword argument together with the ``lr_scheduler_kwargs`` keyword argument to specify arguments
# for the learning rate scheduler as in:
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    lr_scheduler="ExponentialLR",
    lr_scheduler_kwargs={
        "gamma": 0.99,
    },
)
pipeline_result.save_to_directory("nations_transe")

# %%
# Deeper Configuration
# ---------------------
# Arguments for the model can be given as a dictionary using ``model_kwargs``.
pipeline_result = pipeline(
    dataset="Nations",
    model="TransE",
    model_kwargs={
        "scoring_fct_norm": 2,
    },
)
pipeline_result.save_to_directory("nations_transe")

# %%
# The entries in ``model_kwargs`` correspond to the arguments given to ``TransE.__init__``. For a complete listing
# of models, see :mod:`pykeen.models`, where there are links to the reference for each model that explain what
# kwargs are possible. Each model's default hyper-parameters were chosen based on the best reported values from the
# paper originally publishing the model unless otherwise noted on the model's reference page.
#
# Because the pipeline takes care of looking up classes and instantiating them, there are several other parameters
# to :func:`~pykeen.pipeline.pipeline` that can be used to specify the parameters during their respective
# instantiations.
#
# Arguments can be given to the dataset with ``dataset_kwargs``. These are passed on to the
# :class:`~pykeen.datasets.Nations`.
