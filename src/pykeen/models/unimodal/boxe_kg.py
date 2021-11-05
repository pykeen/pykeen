from typing import Any, ClassVar, Mapping, Optional

from torch.nn.init import normal_, uniform_, zeros_
from ...nn.init import uniform_norm_
from ...nn.modules import Interaction
from ...losses import NSSALoss
import torch
from ...models import ERModel
from ...nn.emb import EmbeddingSpecification
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...typing import Hint, Initializer

__all__ = [
    'BoxEKG',
]


SANITY_EPSILON = 10 ** -8


def product_normalise(input_tensor: torch.FloatTensor) -> torch.FloatTensor:
    r"""Normalise the input tensor along its embedding dimension so that the geometric mean is 1

            :param input_tensor: An input tensor with final dimension $d$.
    """
    step1_tensor = torch.abs(input_tensor)  # Compute absolute value of all entries
    step2_tensor = step1_tensor + SANITY_EPSILON    # Prevent zero values by adding a sanity epsilon
    log_norm_tensor = torch.log(step2_tensor)   # Compute the log prior to computing the geom. mean
    step3_tensor = torch.mean(log_norm_tensor, dim=-1, keepdim=True)
    norm_volume = torch.exp(step3_tensor)
    pre_norm_out = input_tensor / norm_volume
    return pre_norm_out


def compute_box(base: torch.FloatTensor, delta: torch.FloatTensor, size: torch.FloatTensor) \
        -> (torch.FloatTensor, torch.FloatTensor):
    r"""Given sets of embeddings of base position, shape, and size, compute the lower and upper corners of
    the resulting box
            :param base: The base position (box center) of the input relation embeddings.
            :param delta: The base shape of the input relation embeddings.
            :param size: The size scalar vectors of the input relation embeddings.
    """
    size_pos = torch.nn.functional.elu(size) + 1  # Enforce that sizes are strictly positive by passing through ELU
    delta_norm = product_normalise(delta)   # Shape vector is normalized using the above helper function
    delta_final = torch.multiply(size_pos, delta_norm)   # Size is learned separately and applied to normalized shape
    # Product normalize the delta
    first_bound = base - 0.5 * delta_final    # Compute potential boundaries by applying the shape in substraction
    second_bound = base + 0.5 * delta_final   # and in addition
    box_low = torch.minimum(first_bound, second_bound)    # Compute box upper bounds using min and max respectively.
    box_high = torch.maximum(first_bound, second_bound)
    return box_low, box_high


def point_to_box_distance(points: torch.FloatTensor, box_lows: torch.FloatTensor, box_highs: torch.FloatTensor) \
        -> torch.FloatTensor:
    r"""Computes the point to box distance function proposed in the BoxE paper in an element-wise fashion.
                :param points: the positions of the points being scored against boxes
                :param box_lows: the lower corners of the boxes
                :param box_highs: the upper corners of the boxes.

         .. math::
            points: p
            box_lows: l
            box_highs: h

            w = h - l . Width is the difference between the upper and lower box bound
            c = (h + l) / 2. Box centers (the mean of the box bounds)

            dist(p,l,h) = |p-c|/(w+1) if l <= p <+ h, |p-c|*(w+1) - 0.5*w*((w+1)-1/(w+1)) otherwise.
    """
    widths = box_highs - box_lows
    widths_p1 = widths + 1  # Compute width plus 1
    centres = 0.5 * (box_lows + box_highs)  # Compute box midpoints
    dist = torch.where(torch.logical_and(points >= box_lows, points <= box_highs),
                       torch.abs(points - centres) / widths_p1,  # If true (inside the box)
                       widths_p1 * torch.abs(points - centres) - (0.5 * widths) * (widths_p1 - 1 / widths_p1))
    return dist


class BoxEKGInteraction(Interaction):
    relation_shape = ('d', 'd', 's', 'd', 'd', 's')  # Boxes are 2xd (size) each, x 2 sets of boxes: head and tail
    entity_shape = ('d', 'd')   # Base position and bump

    def __init__(self, tanh_map: bool = True, norm_order: int = 2):
        r"""Implements the basic BoxE-KG interaction
            :param tanh_map: A Boolean value specifying whether a hyperbolic tangent applies to all representations
            prior to model scoring (default: True)
            :param norm_order: An integer specifying the normalization order (default 2)

                 .. math::
                    points: p
                    box_lows: l
                    box_highs: h

                    w = h - l . Width is the difference between the upper and lower box bound
                    c = (h + l) / 2. Box centers (the mean of the box bounds)

                    dist(p,l,h) = |p-c|/(w+1) if l <= p <+ h, |p-c|*(w+1) - 0.5*w*((w+1)-1/(w+1)) otherwise.
        """
        super().__init__()
        self.tanh_map = tanh_map  # Map the tanh map
        self.norm_order = norm_order

    def forward(self, h, r, t):
        rh_base, rh_delta, rh_size, rt_base, rt_delta, rt_size = r
        h_pos, h_bump = h
        t_pos, t_bump = t
        # First, compute the boxes
        rh_low, rh_high = compute_box(rh_base, rh_delta, rh_size)
        rt_low, rt_high = compute_box(rt_base, rt_delta, rt_size)
        # Second, compute bumped entity representations
        # Normalization
        points_h = (h_pos + t_bump)
        points_t = (t_pos + h_bump)
        # Third, optionally apply the tanh transformation
        if self.tanh_map:
            rh_low = torch.tanh(rh_low)
            rh_high = torch.tanh(rh_high)
            rt_low = torch.tanh(rt_low)
            rt_high = torch.tanh(rt_high)

            points_h = torch.tanh(points_h)
            points_t = torch.tanh(points_t)
        # Fourth, compute the dist function output
        dist_h = point_to_box_distance(points_h, rh_low, rh_high)
        dist_t = point_to_box_distance(points_t, rt_low, rt_high)
        # Fifth, compute the norm
        score_h = dist_h.norm(p=self.norm_order, dim=-1)
        score_t = dist_t.norm(p=self.norm_order, dim=-1)
        total_score = score_h + score_t
        #NSSA = NSSALoss(margin=5, adversarial_temperature=0.0, reduction='sum')
        #if total_score.shape[0] == 512:  # Positive facts hard code
            #print("I'm getting the correct loss here!")
            #print(NSSA.forward(pos_scores=-total_score, neg_scores=-100*total_score,
            #                  neg_weights=torch.ones_like(total_score)))
            #print("That's just a test")
        return - total_score    # Because this is inverted in NSSALoss (higher is better)


class BoxEKG(ERModel):
    r"""An implementation of BoxE

    ---
    citation:
        author: Abboud
        year: 2020
        link: https://arxiv.org/abs/2007.06267
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        p=dict(type=int, low=1, high=2),
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 200,
        tanh_map: bool = True,
        norm_order: int = 2,
        entity_initializer: Hint[Initializer] = uniform_norm_,
        entity_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_initializer: Hint[Initializer] = uniform_norm_,    # Has to be scaled as well
        relation_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        relation_size_initializer: Hint[Initializer] = uniform_,  # Has to be scaled as well
        relation_size_initializer_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        r"""Initialize BoxE-KG

        :param embedding_dim: The entity embedding dimension $d$. Defaults to 200. Is usually $d \in [50, 300]$.
        :param tanh_map: Whether to use tanh mapping after BoxE computation. Default - True
        :param norm_order: Norm Order in score computation (Int): Default - 2
        :param entity_initializer: Entity initializer function. Defaults to :func:`pykeen.nn.init.uniform_norm_`
        :param entity_initializer_kwargs: Keyword arguments to be used when calling the entity initializer
        :param relation_initializer: Relation initializer function. Defaults to :func:`pykeen.nn.init.uniform_norm_`
        :param relation_initializer_kwargs: Keyword arguments to be used when calling the relation initializer
        :param relation_size_initializer: Relation initializer function. Defaults to :func:`torch.nn.init.uniform_`
            Defaults to :func:`torch.nn.init.uniform_`
        :param relation_size_initializer_kwargs: Keyword arguments to be used when calling the
            relation matrix initializer
        """

        super().__init__(
            interaction=BoxEKGInteraction,
            interaction_kwargs=dict(norm_order=norm_order, tanh_map=tanh_map),
            entity_representations=[   # Base position
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=entity_initializer,
                    initializer_kwargs=entity_initializer_kwargs
                ),   # Bump
                # entity bias for head
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=entity_initializer,
                    initializer_kwargs=entity_initializer_kwargs
                )
            ],
            relation_representations=[
                # relation position head
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs
                ),
                # relation shape head
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs
                ),
                EmbeddingSpecification(
                    embedding_dim=1,   # Size
                    initializer=relation_size_initializer,
                    initializer_kwargs=relation_size_initializer_kwargs
                ),
                EmbeddingSpecification(  # Tail position
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs
                ),
                # relation shape tail
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=relation_initializer,
                    initializer_kwargs=relation_initializer_kwargs
                ),
                EmbeddingSpecification(
                    embedding_dim=1,  # Tail Size
                    initializer=relation_size_initializer,
                    initializer_kwargs=relation_size_initializer_kwargs
                )
            ],
            **kwargs,
        )