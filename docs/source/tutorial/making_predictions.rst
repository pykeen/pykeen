.. _making_predictions:

Novel Link Prediction
=====================
After training, the interaction model (e.g., TransE, ConvE, RotatE) can assign a score to an arbitrary triple,
whether it appeared during training, testing, or not. In PyKEEN, each is implemented such that the higher the score
(or less negative the score), the more likely a triple is to be true.

However, for most models, these scores do not have obvious statistical interpretations. This has two main consequences:

1. The score for a triple from one model can not be compared to the score for that triple from another model
2. There is no *a priori* minimum score for a triple to be labeled as true, so predictions must be given as
   a prioritization by sorting a set of triples by their respective scores.

After training a model, there are four high-level interfaces for making predictions:

1. :func:`pykeen.models.predict.get_tail_prediction_df` for a given head/relation pair
2. :func:`pykeen.models.predict.get_relation_prediction_df` for a given head/tail pair
3. :func:`pykeen.models.predict.get_head_prediction_df` for a given relation/tail pair
4. :func:`pykeen.models.predict.get_all_prediction_df` for prioritizing links
5. :func:`pykeen.models.predict.predict_triples_df` for computing scores for explicitly provided triples

Scientifically, :func:`pykeen.models.predict.get_all_prediction_df` is the most interesting in a scenario where
predictions could be tested and validated experimentally.

After Training a Model
~~~~~~~~~~~~~~~~~~~~~~
This example shows using the :func:`pykeen.pipeline.pipeline` to train a model
which will already be in memory. Each of the high-level interfaces are exposed through the
model:

>>> from pykeen.pipeline import pipeline
>>> from pykeen.models import predict
>>> # Run the pipeline
>>> result = pipeline(dataset='Nations', model='RotatE')
>>> # save the model
>>> result.save_to_directory('doctests/nations_rotate')
>>> model = result.model
>>> # Predict tails
>>> predicted_tails_df = predict.get_tail_prediction_df(
...     model, 'brazil', 'intergovorgs', triples_factory=result.training,
... )
>>> # Predict relations
>>> predicted_relations_df = predict.get_relation_prediction_df(
...     model, 'brazil', 'uk', triples_factory=result.training,
... )
>>> # Predict heads
>>> predicted_heads_df = predict.get_head_prediction_df(model, 'conferences', 'brazil', triples_factory=result.training)
>>> # Score all triples (memory intensive)
>>> predictions_df = predict.get_all_prediction_df(model, triples_factory=result.training)
>>> # Score top K triples
>>> top_k_predictions_df = predict.get_all_prediction_df(model, k=150, triples_factory=result.training)
>>> # Score a given list of triples
>>> score_df = predict.predict_triples_df(
...     model=model,
...     triples=[('brazil', 'conferences', 'uk'), ('brazil', 'intergovorgs', 'uk')],
...     triples_factory=result.training,
... )

Loading a Model
~~~~~~~~~~~~~~~
This example shows how to reload a previously trained model. The
:meth:`pykeen.pipeline.PipelineResult.save_to_directory` function makes
a file named ``trained_model.pkl``, so we will use the one from the
previous example.

>>> import torch
>>> from pykeen.datasets import get_dataset
>>> model = torch.load('doctests/nations_rotate/trained_model.pkl')
>>> training = get_dataset(dataset="nations").training
>>> # Predict tails
>>> predicted_tails_df = model.get_tail_prediction_df('brazil', 'intergovorgs', triples_factory=training)
>>> # everything else is the same as above

There's an example model available at
https://github.com/pykeen/pykeen/blob/master/notebooks/hello_world/nations_transe/trained_model.pkl
from the "Hello World" notebook for you to try.

Potential Caveats
-----------------
The model is trained on its ability to predict the appropriate tail for a given head/relation pair as well as its
ability to predict the appropriate head for a given relation/tail pair. This means that while the model can
technically predict relations between a given head/tail pair, it must be done with the caveat that it was not
trained for this task.
