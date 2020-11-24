import timeit
from typing import Mapping, Optional, Tuple

import numpy
import pandas
import torch
import tqdm

from pykeen.nn import Interaction
from pykeen.nn.functional import (
    _complex_interaction_complex_native, _complex_interaction_direct,
    _complex_interaction_optimized_broadcasted,
)
from pykeen.nn.modules import ComplExInteraction, _unpack_singletons
from pykeen.typing import HeadRepresentation, RelationRepresentation, TailRepresentation
from pykeen.version import get_git_hash


def _use_case_to_shape(use_case: str, b: int, n: int) -> Tuple[
    Tuple[int, int],
    Tuple[int, int],
    Tuple[int, int],
]:
    if use_case == "hrt":
        return (b, 1), (b, 1), (b, 1)
    elif use_case == "t":
        return (b, 1), (b, 1), (1, n)
    elif use_case == "h":
        return (1, n), (b, 1), (b, 1)
    else:
        raise ValueError


def _generate_hrt(
    prefix_shapes: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    interaction: Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation],
    dim: int,
    additional_dims: Optional[Mapping[str, int]] = None,
) -> Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation]:
    additional_dims = additional_dims or dict()
    additional_dims.setdefault("d", dim)
    return _unpack_singletons(*(
        torch.rand(*prefix_shape, *(additional_dims[s] for s in suffix_shape), requires_grad=True)
        for prefix_shape, suffix_shape in zip(
        prefix_shapes,
        (
            interaction.entity_shape,
            interaction.relation_shape,
            interaction.tail_entity_shape or interaction.entity_shape
        )
    )
    ))


def _get_result_shape(prefix_shapes) -> Tuple[int, int, int, int]:
    return (max(s[0] for s in prefix_shapes),) + tuple([s[1] for s in prefix_shapes])


def main():
    base_interaction: Interaction = ComplExInteraction()
    variants = [
        _complex_interaction_complex_native,
        _complex_interaction_optimized_broadcasted,
        _complex_interaction_direct,
    ]
    use_case_labels = ["hrt", "t", "h"]
    batch_sizes = [2 ** i for i in range(5, 10 + 1)]
    # batch_sizes = [2 ** i for i in range(5, 7)]
    num_entities = (100, 15_000)
    # num_entities = (100,)
    max_result_elements = 2 ** 30
    vector_dimensions = [2 ** i for i in range(5, 10 + 1)]
    # vector_dimensions = [2 ** i for i in range(5, 7)]
    data = []
    tasks = [
        (b, n, d, ul, _use_case_to_shape(use_case=ul, b=b, n=n))
        for ul in use_case_labels
        for b in batch_sizes
        for n in num_entities
        for d in vector_dimensions
    ]
    progress = tqdm.tqdm(variants, unit="variant", unit_scale=True)
    for variant in progress:
        # create variant
        base_interaction.__class__.func = variant
        for (b, n, d, ul, prefix_shapes) in tqdm.tqdm(tasks, unit="task", unit_scale=True):
            result_shape = _get_result_shape(prefix_shapes)
            n_samples, total_time, time_per_sample = 0, float('nan'), float('nan')
            if max_result_elements is None or numpy.prod(result_shape) < max_result_elements:
                h, r, t = _generate_hrt(prefix_shapes=prefix_shapes, interaction=base_interaction, dim=d)
                try:
                    # TODO: cuda sync
                    timer = timeit.Timer(
                        stmt="interaction(h=h, r=r, t=t)",
                        globals=dict(interaction=base_interaction, h=h, r=r, t=t),
                    )
                    n_samples, total_time = timer.autorange()
                    time_per_sample = total_time / n_samples
                except Exception as error:
                    progress.write(f"ERROR: {error}")
            progress.set_postfix(dict(shape=prefix_shapes, time=time_per_sample))
            data.append((b, n, d, prefix_shapes, ul, variant.__name__, total_time, n_samples, time_per_sample))
    df = pandas.DataFrame(data=data, columns=[
        "batch_size",
        "num_entities",
        "dimension",
        "prefix_shapes",
        "use_case",
        "variant",
        "total_time",
        "n_samples",
        "time_per_sample",
    ])
    git_hash = get_git_hash()
    df.to_csv(f"{git_hash}_measurement.tsv", sep="\t", index=False)
    df_agg = df.groupby(
        by=["batch_size", "num_entities", "dimension", "use_case", "variant"]
    ).agg({"time_per_sample": "mean"}).unstack()
    df_agg.to_csv(f"{git_hash}_measurement_agg.tsv", sep="\t", index=False)
    print(df_agg)


if __name__ == '__main__':
    main()
