Novel Link Prediction
=====================
After training, the interaction model (e.g., TransE, ConvE, RotatE) can assign a score to an arbitrary triple,
whether it appeared during training, testing, or not. In PyKEEN, each is implemented such that the higher the score
(or less negative the score), the more likely a triple is to be true.

However, for most models, these scores do not have obvious statistical interpretations. This has two main consequences:

1. The score for a triple from one model can not be compared to the score for that triple from another model
2. There is no *a priori* minimum score for a triple to be labeled as true, so predictions must be given as
   a prioritization by sorting a set of triples by their respective scores.

After training a model, there are three high-level interfaces for making predictions:

1. :func:`pykeen.models.Model.predict_tails` for a given head/relation pair
2. :func:`pykeen.models.Model.predict_heads` for a given relation/tail pair
3. :func:`pykeen.models.Model.score_all_triples` for prioritizing links

Scientifically, :func:`pykeen.models.Model.score_all_triples` is the most interesting in a scenario where
predictions could be tested and validated experimentally.

.. code-block:: python

    from pykeen.pipeline import pipeline
    results = pipeline(dataset='Nations', model='RotatE')
    model = results.model

    # Predict tails
    predicted_tails_df = model.predict_tails('brazil', 'intergovorgs')

    # Predict heads
    predicted_heads_df = model.predict_heads('conferences', 'brazil')

    # Score All triples
    predictions_df = model.score_all_triples()


Potential Caveats
-----------------
The model is trained on its ability to predict the appropriate tail for a given head/relation pair as well as its
ability to predict the appropriate head for a given relation/tail pair. This means that while the model can
technically predict relations between a given head/tail pair, it must be done with the caveat that it was not
trained for this task.
