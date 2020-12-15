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
    _complex_interaction_optimized_broadcasted,
)
from pykeen.nn.modules import _unpack_singletons
from pykeen.typing import HeadRepresentation, RelationRepresentation, TailRepresentation
from pykeen.version import get_git_hash


def _use_case_to_shape(
    use_case: str,
    b: int,
    n: int,
    num_neg_samples: int,
) -> Tuple[
    Tuple[int, int],
    Tuple[int, int],
    Tuple[int, int],
]:
    if use_case == "hrt":
        b = b * num_neg_samples
        return (b, 1), (b, 1), (b, 1)
    elif use_case == "hrt+":
        return (b, 1), (b, 1), (b, num_neg_samples)
    elif use_case == "h+rt":
        return (b, num_neg_samples), (b, 1), (b, 1)
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
    device: torch.device,
    additional_dims: Optional[Mapping[str, int]] = None,
) -> Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation]:
    additional_dims = additional_dims or dict()
    additional_dims.setdefault("d", dim)
    return _unpack_singletons(*(
        torch.rand(*prefix_shape, *(additional_dims[s] for s in suffix_shape), requires_grad=True, device=device)
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


@click.command()
@click.option('-m', '--max-result-elements-power', type=int, default=30, show_default=True)
@click.option('-b', '--max-batch-size-power', type=int, default=10, show_default=True)
@click.option('-d', '--max-vector-dimension-power', type=int, default=10, show_default=True)
def main(
    max_result_elements_power: int,
    max_batch_size_power: int,
    max_vector_dimension_power: int,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running on {device}.")
    variants = [
        _complex_interaction_complex_native,
        _complex_interaction_optimized_broadcasted,
        _complex_interaction_direct,
    ]
    use_case_labels = ["hrt", "t", "h"]
    batch_sizes = [2 ** i for i in range(5, max_batch_size_power + 1)]
    num_entities = (100, 15_000)
    # num_entities = (100,)
    max_result_elements = 2 ** max_result_elements_power
    vector_dimensions = [2 ** i for i in range(5, max_vector_dimension_power + 1)]
    data = []
    tasks = [
        (b, n, d, ul, _use_case_to_shape(use_case=ul, b=b, n=n))
        for ul in use_case_labels
        for b in batch_sizes
        for n in num_entities
        for d in vector_dimensions
    ]
    progress = tqdm(variants, unit="variant")
    for variant in progress:
        # create variant
        interaction = Interaction.from_func(variant)
        for i, (b, n, d, ul, prefix_shapes) in enumerate(tqdm(tasks, unit="task"), start=1):
            result_shape = _get_result_shape(prefix_shapes)
            n_samples, total_time, time_per_sample = 0, float('nan'), float('nan')
            if max_result_elements is not None and max_result_elements < numpy.prod(result_shape):
                continue
            h, r, t = _generate_hrt(
                prefix_shapes=prefix_shapes,
                interaction=interaction,
                dim=d,
                device=device,
            )
            try:
                timer = Timer(
                    stmt="interaction(h=h, r=r, t=t)",
                    globals=dict(interaction=interaction, h=h, r=r, t=t),
                )
                n_samples, total_time = timer.autorange()
                time_per_sample = total_time / n_samples
            except Exception as error:
                progress.write(f"ERROR: {error}")
            progress.set_postfix(dict(shape=prefix_shapes, time=time_per_sample))
            data.append((i, b, n, d, prefix_shapes, ul, variant.__name__, total_time, n_samples, time_per_sample))

    git_hash = get_git_hash()
    df = pandas.DataFrame(data=data, columns=[
        "experiment_number",
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
    df["device"] = device.type
    df.to_csv(f"{git_hash}_measurement.tsv", sep="\t", index=False)

    df_agg = df.groupby(
        by=["batch_size", "num_entities", "dimension", "use_case", "variant"]
    ).agg({"time_per_sample": "mean"}).unstack().reset_index().dropna()
    df_agg.to_csv(f"{git_hash}_measurement_agg.tsv", sep="\t", index=False)
    print(df_agg)

    viz_df = df[df['total_time'].notna()]
    viz_df['variant'] = viz_df['variant'].map(lambda s: ' '.join(s.split('_')[3:]).capitalize())
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=viz_df, x='experiment_number', y='total_time', hue='variant')
    plt.savefig(f"{git_hash}_measurement.png", dpi=300)


if __name__ == '__main__':
    main()
