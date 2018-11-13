Overview
--------

Knowledge Graph Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~

Knowledge graphs (KGs) are multi-relational, directed graphs in which nodes represent entities and edges represent their
relations (Bordes *et al.* 2013). While they have been successfully applied for question answering, information
extraction, and named entity disambiguation outside of the biomedical domain, their usage in biomedical applications
remains limited (Malone *et al.* 2018; Nickel *et al.*, 2016).

Because KGs are inherently incomplete and noisy, several methods have been developed for deriving or predicting missing
edges (Nickel *et al.*, 2016).  One is to apply reasoning based on formal logic to derive missing edges, but it usually
requires a large set of user-defined formulas to achieve generalization. Another is to train knowledge graph embedding
(KGE) models, which encode the nodes and relations in a KG into a low-dimensional, continuous vector-space that best
preserves the structural characteristics of the KG (Wang *et al.*., 2017). These embeddings can be used to predict new
relations between entities. In a biological setting, relation prediction not only enables researchers to expand their
KGs, but also to generate new hypotheses that can be
tested experimentally.


Existing Software
~~~~~~~~~~~~~~~~~

While there exists other toolkits like OpenKE (Han *et al.*, 2018) and scikit-kge (https://github.com/mnick/scikit-kge),
they are not specifically for bioinformatics applications and they require more expertise in programming and in KGEs.
To the best of our knowledge, BioKEEN is the first framework specifically designed to facilitate the use of KGE models
 for users in the bioinformatics community.

References
~~~~~~~~~~

- Bordes, A., *et al.* (2013). Translating embeddings for modeling multi-relational data. NIPS..
- Wang, Z., *et al.* (2014). Knowledge Graph Embedding by Translating on Hyperplanes. AAAI. Vol. 14.
- Nickel, M., *et al.* (2011) A Three-Way Model for Collective Learning on Multi-Relational Data. ICML. Vol. 11.
- Malone, B., *et al.* (2018). Knowledge Graph Completion to Predict Polypharmacy Side Effects. arXiv preprint arXiv:1810.09227