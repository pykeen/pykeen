# -*- coding: utf-8 -*-

"""Countries dataset."""

from pykeen.datasets.base import UnpackedRemoteDataset

__all__ = [
    'Countries',
]

BASE_URL = 'https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/Countries/Countries_S1/'


class Countries(UnpackedRemoteDataset):
    """The Countries dataset."""

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the Countries small dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        # GitHub's raw.githubusercontent.com service rejects requests that are streamable. This is
        # normally the default for all of PyKEEN's remote datasets, so just switch the default here.
        kwargs.setdefault('stream', False)
        super().__init__(
            training_url=f'{BASE_URL}/train.txt',
            testing_url=f'{BASE_URL}/test.txt',
            validation_url=f'{BASE_URL}/valid.txt',
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


def _main(trials: int = 5):
    from pykeen.pipeline import pipeline
    from tabulate import tabulate
    from tqdm.contrib.itertools import product
    import pandas as pd
    from pykeen.constants import PYKEEN_EXPERIMENTS

    datasets = [Countries]
    models = ['TransE', 'ComplEx', 'RotatE']
    frequencies = [1, 2, 5, 10]
    patiences = [3, 5, 7]
    deltas = [0.001, 0.002, 0.02]

    rows = []

    it = product(datasets, models, range(trials), frequencies, patiences, deltas, desc='Early Stopper HPO')
    for dataset, model, trial, frequency, patience, relative_delta in it:
        results = pipeline(
            dataset=dataset,
            model=model,
            random_seed=trial,
            device='cpu',
            stopper='early',
            stopper_kwargs=dict(
                metric='adjusted_mean_rank',
                frequency=frequency,
                patience=patience,
                relative_delta=relative_delta,
            ),
            training_kwargs=dict(num_epochs=1000),
            evaluation_kwargs=dict(use_tqdm=False),
            automatic_memory_optimization=False,  # not necessary on CPU
        )
        rows.append((
            dataset if isinstance(dataset, str) else dataset.get_normalized_name(),
            model,
            trial,
            frequency,
            patience,
            relative_delta,
            len(results.losses),
            results.metric_results.get_metric('both.avg.adjusted_mean_rank'),
            results.metric_results.get_metric('hits@10'),
        ))

    df = pd.DataFrame(rows, columns=[
        'Dataset', 'Model', 'Trial', 'Frequency', 'Patience', 'Delta', 'Epochs', 'AMR', 'Hits@10',
    ])
    df.to_csv(PYKEEN_EXPERIMENTS / 'stopping_hpo.tsv', sep='\t', index=False)
    print(tabulate(df, headers=df.columns))


if __name__ == '__main__':
    _main()
