import itertools
import pathlib
from typing import Callable, Iterable

import click
import pandas
import seaborn
import torch
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


def _iter_expected_ranks(
    transformation: Callable[[torch.Tensor], torch.Tensor],
    num_entities: int,
    num_triples: int,
    num_samples: int,
) -> Iterable[float]:
    # individual ranks
    r = 1 + torch.arange(num_entities, dtype=torch.float64)
    # transformation
    r = transformation(r)
    # sample
    for _ in range(num_samples):
        yield r[
            torch.randint(num_entities, size=(num_triples,))
        ].mean().reciprocal().item()


@click.command()
# @click.option("-e", "--num-entities", type=int, default=15_000)
# @click.option("-t", "--num-triples", type=int, default=300_000)
@click.option("-s", "--num-samples", type=int, default=1_000)
def main(
    num_samples: int,
):
    """Main entry point."""
    buffer_path = pathlib.Path("/tmp/hmr.tsv")
    if buffer_path.is_file():
        df = pandas.read_csv(buffer_path, sep="\t")
    else:
        df = pandas.DataFrame(
            data=list(
                tqdm(
                    (
                        (num_triples, num_entities, value)
                        for num_triples, num_entities in itertools.product(
                            (2 ** i for i in range(5, 16)),
                            (2 ** i for i in range(5, 16)),
                        )
                        for value in tqdm(
                            _iter_expected_ranks(
                                transformation=torch.FloatTensor.reciprocal,
                                num_entities=num_entities,
                                num_triples=num_triples,
                                num_samples=num_samples,
                            ),
                            unit="sample",
                            unit_scale=True,
                            leave=False,
                        )
                    ),
                    unit="combination",
                    unit_scale=True,
                )
            ),
            columns=["num_triples", "num_entities", "value"],
        )
        df.to_csv(buffer_path, sep="\t", index=False)

    facet: seaborn.FacetGrid = seaborn.relplot(
        data=df,
        x="num_entities",
        y="value",
        kind="line",
        hue="num_triples",
        ci="sd",
        legend="full",
    )
    facet.set(
        xscale="log",
        yscale="log",
    )
    plt.show()
    # print(df.groupby(by=["num_triples", "num_entities"]).agg({"value": ["mean", "std"]}))


if __name__ == "__main__":
    main()
