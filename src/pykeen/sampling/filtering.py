# -*- coding: utf-8 -*-

r"""Consider the following properties of relation $r$. Because the corruption operations (see `Corruption`_)
are applied independently of triples, the resulting candidate corrupt triples could overlap with known positive
triples in $\mathcal{K}$.

=====================  ============================================  ==============================================================
Property of :math:`r`  Example pair of triples                       Implications
=====================  ============================================  ==============================================================
one-to-many            :math:`(h,r,t_1), (h,r,t_2) \in \mathcal{K}`  :math:`(h,r,t_2) \in T(h,r,t_1) \cup (h,r,t_1) \in T(h,r,t_2)`
multiple               :math:`(h,r_1,t), (h,r_2,t) \in \mathcal{K}`  :math:`(h,r_2,t) \in R(h,r_1,t) \cup (h,r_1,t) \in R(h,r_2,t)`
many-to-one            :math:`(h_1,r,t), (h_2,r,t) \in \mathcal{K}`  :math:`(h_2,r,t) \in H(h_1,r,t) \cup (h_1,r,t) \in H(h_2,r,t)`
=====================  ============================================  ==============================================================

If no relations in $\mathcal{K}$ satisfy any of the relevant properties for the corruption schema chosen in negative
sampling, then there is guaranteed to be no overlap between $\mathcal{N}$ and $\mathcal{K}$ such that 
$\mathcal{N} \cap \mathcal{K} \neq \emptyset$. However, this scenario is very unlikely for real-world knowledge graphs.

The known positive triples that appear in $\mathcal{N}$ are false negatives. This is problematic becuase they will
be scored well by the knowledge graph embedding model during evaluation, have lower ranks, and ultimately lead to
worse performance on rank-based evaluation metrics such as the (arithmetic) mean rank.

Identifying False Negatives
---------------------------
[bordes2013]_ proposed an exact algorithm in which all known positive triples in $\mathcal{K}$ are excluded from
the set of candidate negative triples $\mathcal{N}$ such that $\mathcal{N}^- = \mathcal{N} \setminus \mathcal{K}$
in order to yield more accurate evaluations. However, in practice, $|\mathcal{N}| \gg |\mathcal{K}|$, so the
likelihood of generating a false negative is rather low. Therefore, the additional filter step is often omitted
to lower computational cost.

By default, PyKEEN does *not* filter false negatives from $\mathcal{N}$. To enable the "filtered setting", the
``filtered`` keyword can be given to ``negative_sampler_kwargs`` like in:

.. code-block:: python

    results = pipeline(
        dataset='YAGO3-10',
        model='PairRE',
        training_loop='sLCWA',
        negative_sampler='basic',
        negative_sampler_kwargs=dict(
            filtered=True,    
        ),
    )

PyKEEN implements several algorithms for filtering with different properties that can be chosen using the
``filterer`` keyword argument in ``negative_sampler_kwargs``. By default, an exact algorithm is used in
:class:`pykeen.sampling.filtering.DefaultFilterer`. However, a filterer based on
`bloom filters <https://en.wikipedia.org/wiki/Bloom_filter>`_ is also available in
:class:`pykeen.sampling.filtering.BloomFilterer` that trades exact correctness for speed and efficiency.
It can be activated with:

.. code-block:: python

    from pykeen.pipeline import pipeline

    results = pipeline(
        dataset='YAGO3-10',
        model='PairRE',
        training_loop='sLCWA',
        negative_sampler='basic',
        negative_sampler_kwargs=dict(
            filtered=True,
            filterer='bloom',    
        ),
    )

More information on them can be found in :mod:`pykeen.sampling.filtering`.

.. warning:: 

    It should be taken into account that a corrupted triple that is *not part*
    of the knowledge graph can represent a true fact. These false negatives can
    not be removed *a priori* in the filtered setting because they are unknown.
"""  # noqa

import math
from abc import abstractmethod
from typing import Iterable, Optional, Tuple

import torch
from class_resolver import Resolver
from torch import nn

from ..triples import CoreTriplesFactory

__all__ = [
    "filterer_resolver",
    "Filterer",
    "DefaultFilterer",
    "BloomFilterer",
    "PythonSetFilterer",
]


class Filterer(nn.Module):
    """An interface for filtering methods for negative triples."""

    @abstractmethod
    def forward(
        self,
        negative_batch: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, Optional[torch.BoolTensor]]:
        """Filter all proposed negative samples that are positive in the training dataset.

        Normally there is a low probability that proposed negative samples are positive in the training datasets and
        thus act as false negatives. This is expected to act as a kind of regularization, since it adds noise signal to
        the training data. However, the degree of regularization is hard to control since the added noise signal depends
        on the ratio of true triples for a given entity relation or entity entity pair. Therefore, the effects are hard
        to control and a researcher might want to exclude the possibility of having false negatives in the proposed
        negative triples.

        .. note ::
            Filtering is a very expensive task, since every proposed negative sample has to be checked against the
            entire training dataset.

        :param negative_batch: shape: ???
            The batch of negative triples.

        :return:
            A pair (filtered_negative_batch, keep_mask) of shape ???
        """
        raise NotImplementedError


class DefaultFilterer(Filterer):
    """The default filterer.

    .. warning:: This filterer may contain a correctness error, cf. https://github.com/pykeen/pykeen/issues/272
    """

    def __init__(self, triples_factory: CoreTriplesFactory):
        """Initialize the filterer.

        :param triples_factory: The triples factory.
        """
        super().__init__()
        # Make sure the mapped triples are initiated
        # Copy the mapped triples to the device for efficient filtering
        self.register_buffer(name="mapped_triples", tensor=triples_factory.mapped_triples)

    def forward(
        self,
        negative_batch: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, Optional[torch.BoolTensor]]:  # noqa: D102
        try:
            # Check which heads of the mapped triples are also in the negative triples
            head_filter = (
                self.mapped_triples[:, 0:1].view(1, -1) == negative_batch[:, 0:1]  # type: ignore
            ).max(axis=0)[0]
            # Reduce the search space by only using possible matches that at least contain the head we look for
            sub_mapped_triples = self.mapped_triples[head_filter]  # type: ignore
            # Check in this subspace which relations of the mapped triples are also in the negative triples
            relation_filter = (sub_mapped_triples[:, 1:2].view(1, -1) == negative_batch[:, 1:2]).max(axis=0)[0]
            # Reduce the search space by only using possible matches that at least contain head and relation we look for
            sub_mapped_triples = sub_mapped_triples[relation_filter]
            # Create a filter indicating which of the proposed negative triples are positive in the training dataset
            final_filter = (sub_mapped_triples[:, 2:3].view(1, -1) == negative_batch[:, 2:3]).max(axis=1)[0]
        except RuntimeError as e:
            # In cases where no triples should be filtered, the subspace reduction technique above will fail
            if str(e) == (
                'cannot perform reduction function max on tensor with no elements because the operation does not '
                'have an identity'
            ):
                final_filter = torch.zeros(negative_batch.shape[0], dtype=torch.bool, device=negative_batch.device)
            else:
                raise e
        # Return only those proposed negative triples that are not positive in the training dataset
        return negative_batch[~final_filter], ~final_filter


class PythonSetFilterer(Filterer):
    """A filterer using Python sets for filtering.

    This filterer is expected to be rather slow due to the conversion from torch long tensors to Python tuples. It can
    still serve as a baseline for performance comparison.
    """

    def __init__(self, triples_factory: CoreTriplesFactory):
        """Initialize the filterer.

        :param triples_factory: The triples factory.
        """
        super().__init__()
        # store set of triples
        self.triples = set(map(tuple, triples_factory.mapped_triples.tolist()))

    def forward(
        self,
        negative_batch: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, Optional[torch.BoolTensor]]:  # noqa: D102
        keep_mask = torch.as_tensor(
            data=[tuple(triple) not in self.triples for triple in negative_batch.tolist()],
            dtype=torch.bool,
        )
        return negative_batch[keep_mask], keep_mask


class BloomFilterer(Filterer):
    """A filterer for negative triples based on the Bloom filter.

    Pure PyTorch, a proper module which can be moved to GPU, and support batch-wise computation.

    .. seealso ::
        * https://github.com/hiway/python-bloom-filter/ - for calculation of sizes, and rough structure of code
        * https://github.com/skeeto/hash-prospector#two-round-functions - for parts of the hash function
    """

    #: some prime numbers for tuple hashing
    mersenne: torch.LongTensor

    #: The bit-array for the Bloom filter data structure
    bit_array: torch.BoolTensor

    def __init__(self, triples_factory: CoreTriplesFactory, error_rate: float = 0.001):
        """Initialize the Bloom filter based filterer.

        :param triples_factory: The triples factory.
        :param error_rate: The desired error rate.
        """
        super().__init__()

        # Allocate bit array
        self.ideal_num_elements = triples_factory.num_triples
        size = self.num_bits(num=self.ideal_num_elements, error_rate=error_rate)
        self.register_buffer(name="bit_array", tensor=torch.zeros(size, dtype=torch.bool))
        self.register_buffer(
            name="mersenne",
            tensor=torch.as_tensor(
                data=[2 ** x - 1 for x in [17, 19, 31]],
                dtype=torch.long,
            ).unsqueeze(dim=0),
        )

        # calculate number of hashing rounds
        self.rounds = self.num_probes(num_elements=self.ideal_num_elements, num_bits=size)

        # index triples
        self.add(triples=triples_factory.mapped_triples)

        # Store some meta-data
        self.error_rate = error_rate

    def __repr__(self):  # noqa:D105
        return (
            f"{self.__class__.__name__}("
            f"error_rate={self.error_rate}, "
            f"size={self.bit_array.shape[0]}, "
            f"rounds={self.rounds}, "
            f"ideal_num_elements={self.ideal_num_elements}, "
            f")"
        )

    @staticmethod
    def num_bits(num: int, error_rate: float = 0.01) -> int:
        """
        Determine the required number of bits.

        :param num:
            The number of elements the Bloom filter shall store.
        :param error_rate:
            The desired error rate.

        :return:
            The required number of bits.
        """
        numerator = -1 * num * math.log(error_rate)
        denominator = math.log(2) ** 2
        real_num_bits_m = numerator / denominator
        return int(math.ceil(real_num_bits_m))

    @staticmethod
    def num_probes(num_elements: int, num_bits: int):
        """
        Determine the number of probes / hashing rounds.

        :param num_elements:
            The number of elements.
        :param num_bits:
            The number of bits, i.e., the size of the Bloom filter.

        :return:
            The number of hashing rounds.
        """
        num_bits = num_bits
        real_num_probes_k = (num_bits / num_elements) * math.log(2)
        return int(math.ceil(real_num_probes_k))

    def probe(
        self,
        batch: torch.LongTensor,
    ) -> Iterable[torch.LongTensor]:
        """
        Iterate over indices from the probes.

        :param batch: shape: (batch_size, 3)
            A batch of elements.

        :yields:
            Indices of the k-th round, shape: (batch_size,).
        """
        # pre-hash
        x = (self.mersenne * batch).sum(dim=-1)
        for _ in range(self.rounds):
            # cf. https://github.com/skeeto/hash-prospector#two-round-functions
            x = x ^ (x >> 16)
            x = x * 0x7feb352d
            x = x ^ (x >> 15)
            x = x * 0x846ca68b
            x = x ^ (x >> 16)
            yield x % self.bit_array.shape[0]

    def add(self, triples: torch.LongTensor) -> None:
        """
        Add triples to the Bloom filter.

        :param triples:
            The triples.
        """
        for i in self.probe(batch=triples):
            self.bit_array[i] = True

    def contains(self, batch: torch.LongTensor) -> torch.BoolTensor:
        """
        Check whether a triple is contained.

        :param batch: shape (batch_size, 3)
            The batch of triples.

        :return: shape: (batch_size,)
            The result. False guarantees that the element was not contained in the indexed triples. True can be
            erroneous.
        """
        result = batch.new_ones(batch.shape[0], dtype=torch.bool)
        for i in self.probe(batch):
            result &= self.bit_array[i]
        return result

    def forward(
        self,
        negative_batch: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, Optional[torch.BoolTensor]]:  # noqa: D102
        keep_mask = ~self.contains(batch=negative_batch)
        return negative_batch[keep_mask], keep_mask


filterer_resolver = Resolver.from_subclasses(
    base=Filterer,
    default=DefaultFilterer,
)
