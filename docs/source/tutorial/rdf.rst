Learning from RDF
=================

Resource Description framework (RDF) is a typical way of storing
subject-predicate-object triples, which is often directly amenable
to knowledge graph embedding models.


.. code-block:: python

    import rdflib
    from pykeen.triples import TriplesFactory

    graph = rdflib.Graph()
    graph.parse(...)
    triples_factory = TriplesFactory.from_rdflib(graph)

Some caveats when learning on RDF:

1. RDF doesn't necessarily represent a logical knowledge graph -
   it might be littered with blank nodes and other constructions
   that represent metadata on triples
2. RDF can contain literals for objects, these are excluded from
   the canned workflow