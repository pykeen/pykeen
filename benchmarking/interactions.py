import random
from typing import Mapping, Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn as sns
import torch
from torch.utils.benchmark import Timer
from tqdm import tqdm

from pykeen.nn import Interaction
from pykeen.nn.functional import (
    _complex_interaction_complex_native, _complex_interaction_direct,
    _complex_interaction_optimized_broadcasted, _complex_select,
)
from pykeen.typing import HeadRepresentation, RelationRepresentation, TailRepresentation
from pykeen.utils import unpack_singletons
from pykeen.version import get_git_hash


def _use_case_to_shape(
    use_case: str,
    b: int,
    n: int,
    s: int,
) -> Tuple[
    Tuple[int, int],
    Tuple[int, int],
    Tuple[int, int],
]:
    """
    Generate prefix shapes for various use cases.

    :param use_case:
        The use case.

            - "hrt": score_hrt naive
            - "hrt+": score_hrt fast SCLWA with tail corruption
            - "h+rt": score_hrt fast SCLWA with head corruption
            - "t": score_t
            - "h": score_t

    :param b:
        The batch size.
    :param n:
        The number of entities.
    :param s:
        The number of negative samples.

    :return:
        A 3-tuple, (head_prefix, relation_prefix, tail_prefix), each a 2-tuple of integers.
    """
    if use_case == "hrt":
        b = b * s
        return (b, 1), (b, 1), (b, 1)
    elif use_case == "hrt+":
        return (b, 1), (b, 1), (b, s)
    elif use_case == "h+rt":
        return (b, s), (b, 1), (b, 1)
    elif use_case == "t":
        return (b, 1), (b, 1), (1, n)
    elif use_case == "h":
        return (1, n), (b, 1), (b, 1)
    else:
        raise ValueError


def _resolve_shapes(
    prefix_shapes: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    interaction: Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation],
    dim: int,
    additional_dims: Optional[Mapping[str, int]] = None,
) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    additional_dims = additional_dims or dict()
    additional_dims.setdefault("d", dim)
    return [
        tuple((*prefix_shape, *(additional_dims[s] for ss in s)) for s in suffix_shape)
        for prefix_shape, suffix_shape in zip(
            prefix_shapes,
            (
                interaction.entity_shape,
                interaction.relation_shape,
                interaction.tail_entity_shape or interaction.entity_shape
            )
        )
    ]


def _generate_hrt(
    shapes: Tuple[Tuple[Tuple[int, ...], ...], ...],
    device: torch.device,
) -> Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation]:
    return unpack_singletons(*(
        [
            torch.rand(*shape, requires_grad=True, device=device)
            for shape in single_shapes
        ]
        for single_shapes in shapes
    ))


def _get_result_shape(prefix_shapes) -> Tuple[int, int, int, int]:
    return (max(s[0] for s in prefix_shapes),) + tuple([s[1] for s in prefix_shapes])


@click.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--shuffle/--no-shuffle', default=False)
@click.option('-m', '--max-result-elements-power', type=int, default=30, show_default=True)
@click.option('-n', '--max-num-entities-power', type=int, default=15, show_default=True)
@click.option('-b', '--max-batch-size-power', type=int, default=10, show_default=True)
@click.option('-d', '--max-vector-dimension-power', type=int, default=10, show_default=True)
@click.option('-s', '--max-sample-power', type=int, default=10, show_default=True)
def main(
    fast: bool,
    shuffle: bool,
    max_result_elements_power: int,
    max_num_entities_power: int,
    max_batch_size_power: int,
    max_vector_dimension_power: int,
    max_sample_power: int,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running on {device}.")
    variants = [
        _complex_select,
        _complex_interaction_complex_native,
        _complex_interaction_optimized_broadcasted,
        _complex_interaction_direct,
    ]
    use_case_labels = ["hrt", "hrt+", "h+rt", "t", "h"]
    batch_sizes = [2 ** i for i in range(5, max_batch_size_power + 1)]
    negative_samples = [2 ** i for i in range(5, max_sample_power + 1)]
    num_entities = [2 ** i for i in range(7, max_num_entities_power)]
    max_result_elements = 2 ** max_result_elements_power
    vector_dimensions = [2 ** i for i in range(5, max_vector_dimension_power + 1)]
    data = []
    tasks = [
        (b, s, n, d, ul, _use_case_to_shape(use_case=ul, b=b, n=n, s=s))
        for b in batch_sizes
        for s in negative_samples
        for n in num_entities
        for d in vector_dimensions
        for ul in use_case_labels
    ]
    if fast:
        tasks = tasks[:5]
    if shuffle:
        random.shuffle(tasks)
    progress = tqdm(variants, unit="variant")
    for variant in progress:
        # create variant
        interaction = Interaction.from_func(variant)
        for i, (b, s, n, d, ul, prefix_shapes) in enumerate(tqdm(tasks, unit="task"), start=1):
            result_shape = _get_result_shape(prefix_shapes)
            median = iqr = float('nan')
            if max_result_elements is not None and max_result_elements < numpy.prod(result_shape):
                continue
            shapes = _resolve_shapes(
                prefix_shapes=prefix_shapes,
                interaction=interaction,
                dim=d,
            )
            try:
                timer = Timer(
                    stmt="interaction(h=h, r=r, t=t)",
                    globals=dict(interaction=interaction, shapes=shapes, device=device, _generate_hrt=_generate_hrt),
                    setup="h, r, t = _generate_hrt(shapes=shapes, device=device)"
                )
                time = timer.blocked_autorange()
                median = time.median
                iqr = time.iqr
            except Exception as error:
                progress.write(f"ERROR: {error}")
            progress.set_postfix(dict(shape=prefix_shapes, time=median))
            data.append((i, b, s, n, d, ul, prefix_shapes, variant.__name__, median, iqr))

    git_hash = get_git_hash()
    df = pandas.DataFrame(data=data, columns=[
        "experiment_number",
        "batch_size",
        "num_negative_samples",
        "num_entities",
        "dimension",
        "use_case",
        "prefix_shapes",
        "variant",
        "time_median",
        "time_inter_quartile_range",
    ])
    df["device"] = device.type
    df.to_csv(f"{git_hash}_measurement.tsv", sep="\t", index=False)

    df_agg = df.groupby(
        by=["batch_size", "num_entities", "dimension", "use_case", "variant"]
    ).agg({"time_median": "mean"}).unstack().reset_index().dropna()
    df_agg.to_csv(f"{git_hash}_measurement_agg.tsv", sep="\t", index=False)
    print(df_agg)

    viz_df = df[df["time_median"].notna()]
    viz_df["variant"] = viz_df["variant"].map(lambda s: ' '.join(s.split('_')[3:]).capitalize())
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=viz_df, x='experiment_number', y='time_median', hue='variant')
    plt.savefig(f"{git_hash}_measurement.png", dpi=300)


if __name__ == '__main__':
    main()
