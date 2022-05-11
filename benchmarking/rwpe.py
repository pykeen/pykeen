"""Benchmarks for random walk positional encoding."""
import itertools
import multiprocessing
import random
from typing import Sequence

import numpy
import pandas
import seaborn
import torch
from torch.utils.benchmark import Compare, Timer
from tqdm.auto import tqdm

from pykeen.nn.utils import sparse_eye


def extract_diagonal_explicit_loop(matrix: torch.Tensor) -> torch.Tensor:
    """Extract diagonal by item access and Python for-loop."""
    n = matrix.shape[0]
    d = torch.zeros(n, device=matrix.device)

    # torch.sparse.coo only allows direct numbers here, can't feed an eye tensor here
    for i in range(n):
        d[i] = matrix[i, i]

    return d


def extract_diagonal_coalesce(matrix: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
    """Extract diagonal using coalesce."""
    n = matrix.shape[0]
    d = torch.zeros(n, device=matrix.device)

    d_sparse = (matrix * eye).coalesce()
    indices = d_sparse.indices()
    values = d_sparse.values()
    d[indices] = values

    return d


def extract_diagonal_manual_coalesce(matrix: torch.Tensor) -> torch.Tensor:
    """Extract diagonal using a manual implementation accessing private functions of an instable API."""
    n = matrix.shape[0]
    d = torch.zeros(n, device=matrix.device)

    indices = matrix._indices()
    mask = indices[0] == indices[1]
    diagonal_values = matrix._values()[mask]
    diagonal_indices = indices[0][mask]

    return d.scatter_add(dim=0, index=diagonal_indices, src=diagonal_values)


def create_sparse_matrix(n: int, density: float) -> torch.Tensor:
    """Create a sparse matrix of the given size and (maximum) density."""
    # TODO: the sparsity pattern may not be very natural for a graph
    nnz = int(n**2 * density)
    indices = torch.randint(n, size=(2, nnz))
    values = torch.ones(nnz)
    return torch.sparse_coo_tensor(indices=indices, values=values, size=(n, n))


def test_extract_diagonal():
    """Test that all three methods yield the same result."""
    n = 16_000
    sparsity = 1.0e-03
    matrix = create_sparse_matrix(n=n, density=sparsity)
    eye = sparse_eye(n=n)
    reference = extract_diagonal_coalesce(matrix=matrix, eye=eye)
    assert torch.allclose(extract_diagonal_explicit_loop(matrix=matrix), reference)
    assert torch.allclose(extract_diagonal_manual_coalesce(matrix=matrix), reference)


class Grid:
    def __init__(self, *iters: Sequence, shuffle: bool = False) -> None:
        self.size = numpy.prod(map(len, iters))
        grid = itertools.product(*iters)
        if shuffle:
            grid = list(grid)
            random.shuffle(grid)
        self.grid = grid

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        yield from self.grid


def time_extract_diagonal(fast: bool = False):
    """Time the different variants."""
    n_cpu = multiprocessing.cpu_count()
    # fb15k237 sparsity ~ 300k / 15_000**2 = 0.001
    density_grid = (1.0e-04, 1.0e-03, 1.0e-02)
    num_threads_grid = (1, n_cpu // 2, n_cpu)
    n_grid = (4_000, 8_000, 16_000, 32_000, 64_000)
    methods = (
        "extract_diagonal_coalesce(matrix=matrix, eye=eye)",
        # "extract_diagonal_explicit_loop(matrix=matrix)",
        "extract_diagonal_manual_coalesce(matrix=matrix)",
    )
    # fast run
    if fast:
        num_threads_grid = (1,)
        n_grid = (8_000,)
        density_grid = (1.0e-02,)
    measurements = []
    data = []
    for stmt, num_threads, n, density in tqdm(
        Grid(
            methods,
            num_threads_grid,
            n_grid,
            density_grid,
            # we shuffle here to have a more consistent time estimate of tqdm
            shuffle=True,
        ),
        unit="configuration",
        unit_scale=True,
    ):
        measurement = Timer(
            stmt=stmt,
            setup=f"matrix=create_sparse_matrix(n={n}, density={density}); eye=sparse_eye(n={n});",
            globals=dict(
                create_sparse_matrix=create_sparse_matrix,
                sparse_eye=sparse_eye,
                extract_diagonal_coalesce=extract_diagonal_coalesce,
                extract_diagonal_explicit_loop=extract_diagonal_explicit_loop,
                extract_diagonal_manual_coalesce=extract_diagonal_manual_coalesce,
            ),
            # description needs to be present
            description=f"n={n}",
            label=f"density={density}",
            num_threads=num_threads,
        ).blocked_autorange()
        print(measurement)
        measurements.append(measurement)
        method = stmt.split("(")[0].replace("extract_diagonal_", "")
        data.append((method, n, density, num_threads, measurement.median, measurement.mean, measurement.iqr))
    comparison = Compare(measurements)
    comparison.colorize()
    comparison.trim_significant_figures()
    comparison.print()
    df = pandas.DataFrame(data=data, columns=["method", "n", "density", "num_threads", "median", "mean", "iqr"])
    df.to_csv("./times.tsv", sep="\t", index=False)
    grid: seaborn.FacetGrid = seaborn.relplot(
        data=df, x="n", y="median", hue="method", size="density", style="num_threads", kind="line",
    )
    grid.set(xscale="log", ylabel="median time [s]")
    grid.tight_layout()
    grid.savefig("./times.svg")


if __name__ == "__main__":
    test_extract_diagonal()
    time_extract_diagonal()
