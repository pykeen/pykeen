"""Benchmark utility methods."""
import functools
import itertools
import operator
import timeit
from typing import Sequence, Tuple

import click
import pandas
import torch
from tqdm.auto import tqdm

from pykeen.utils import tensor_sum


def _generate_shapes(
    batch_size: int,
    num: int,
    dim: int,
    use_case: str,
    shapes: Sequence[str],
) -> Sequence[Tuple[int, ...]]:
    dims = dict(b=batch_size, h=num, r=num, t=num, d=dim)
    canonical = "bhrtd"
    return [
        tuple(
            dims[c] if (
                c in use_case and c in shape
            ) else 1
            for c in canonical
        )
        for shape in shapes
    ]


def _generate_tensors(*shapes: Tuple[int, ...]) -> Sequence[torch.FloatTensor]:
    return [
        torch.rand(shape, requires_grad=True, dtype=torch.float32)
        for shape in shapes
    ]


def tqdm_itertools_product(*args, **kwargs):
    return tqdm(itertools.product(*args), **kwargs, total=functools.reduce(operator.mul, map(len, args), 1))


def _get_result_shape(shapes: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:
    return tuple(max(ds) for ds in zip(*shapes))


@click.command()
@click.option('-m', '--max-result-power', type=int, default=30, show_default=True)
@click.option('-b', '--max-batch-size-power', type=int, default=10, show_default=True)
@click.option('-n', '--max-num-power', type=int, default=14, show_default=True)
@click.option('-d', '--max-dim-power', type=int, default=10, show_default=True)
def main(
    max_result_power: int,
    max_batch_size_power: int,
    max_dim_power: int,
    max_num_power: int,
):
    """Test whether tensor_sum actually offers any performance benefits."""
    max_size = 2 ** max_result_power
    data = []
    progress = tqdm_itertools_product(
        [2 ** i for i in range(5, max_batch_size_power + 1)],
        [2 ** i for i in range(10, max_num_power + 1)],  # 2**15 ~ 15k
        [2 ** i for i in range(5, max_dim_power + 1)],
        ["b", "bh", "br", "bt"],  # score_hrt, score_h, score_t, score_t
        (
            ("ConvKB/ERMLP", "", "bh", "br", "bt"),  # conv_bias, h, r, t
            ("NTN", "bhrt", "bhr", "bht", "br"),  # h w t, vh h, vt t, b
            ("ProjE", "bh", "br", ""),  # h, r, b_c
            # ("RotatE", "bhr", "bt"),  # hr, -t,
            # ("RotatE-inv", "bh", "brt"),  # h, -(r_inv)t,
            ("StructuredEmbedding", "bhr", "brt"),  # r h, r t
            ("TransE/TransD/KG2E", "bh", "br", "bt"),  # h, r, t
            ("TransH", "bh", "bhr", "br", "bt", "brt"),  # h, -<h, w_r> w_r, d_r, -t,, <t, w_r> w_r
            ("TransR", "bhr", "br", "brt"),  # h m_r, r, -t m_r
            # ("UnstructuredModel", "bh", "bt"),  # h, r
        ),
        unit="configuration",
        unit_scale=True,
    )
    for batch_size, num, dim, use_case, (models, *shapes) in progress:
        this_shapes = _generate_shapes(
            batch_size=batch_size,
            num=num,
            dim=dim,
            use_case=use_case,
            shapes=shapes,
        )
        result_shape = _get_result_shape(this_shapes)
        num_result_elements = functools.reduce(operator.mul, result_shape, 1)
        if num_result_elements > max_size:
            continue
        tensors = _generate_tensors(*this_shapes)

        # using normal sum
        n_samples, time_baseline = timeit.Timer(
            stmt="sum(tensors)",
            globals=dict(tensors=tensors)
        ).autorange()
        time_baseline /= n_samples

        # use tensor_sum
        n_samples, time = timeit.Timer(
            setup="tensor_sum(*tensors)",
            stmt="tensor_sum(*tensors)",
            globals=dict(tensor_sum=tensor_sum, tensors=tensors)
        ).autorange()
        time /= n_samples

        data.append((batch_size, num, dim, use_case, shapes, time_baseline, time))
        progress.set_postfix(shape=result_shape, delta=time_baseline - time)
    df = pandas.DataFrame(data=data, columns=[
        "batch_size",
        "num",
        "dim",
        "use_case",
        "shapes",
        "time_sum",
        "time_tensor_sum",
    ])
    df.to_csv("tensor_sum.perf.tsv", sep="\t", index=False)
    print("tensor_sum is better than sum in {percent:2.2%} of all cases.".format(
        percent=(df["time_sum"] > df["time_tensor_sum"]).mean())
    )


if __name__ == '__main__':
    main()
