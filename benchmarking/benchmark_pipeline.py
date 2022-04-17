import json
import pathlib
import pprint
import click
import pandas
import logging
import more_click
from pykeen.constants import PYKEEN_BENCHMARKS
from pykeen.pipeline.api import pipeline_from_config, replicate_pipeline_from_config
from pykeen.utils import (
    CONFIGURATION_FILE_FORMATS,
    format_relative_comparison,
    get_devices,
    load_configuration,
    resolve_device,
)
from pykeen.experiments.cli import HERE
from pykeen.version import get_git_hash

logger = logging.getLogger(__name__)


@click.command()
@click.option("-c", "--configuration-root", type=pathlib.Path, default=HERE)
@click.option("-o", "--output-root", type=pathlib.Path, default=PYKEEN_BENCHMARKS.joinpath("pipeline", get_git_hash()))
@click.option("-e", "--num-epochs", type=int, default=5)
@more_click.log_level_option()
def main(
    configuration_root: pathlib.Path,
    output_root: pathlib.Path,
    num_epochs: int,
    log_level: str,
):
    """"""
    logging.basicConfig(level=log_level)

    device = resolve_device(device=None)
    # TODO: collect info about system?
    logging.info(f"Running on device: {device}")

    configuration_paths = sorted(
        path for ext in CONFIGURATION_FILE_FORMATS for path in configuration_root.rglob(f"*{ext}")
    )
    logger.info(f"Found {len(configuration_paths)} configurations under {configuration_root}")

    for i, path in enumerate(configuration_paths, start=1):
        logger.info(f"Progress: {format_relative_comparison(part=i, total=len(configuration_paths))}")
        reference, model, dataset, *remainder = path.stem.split("_")
        if model in {
            "nodepiece",  # no precomputed anchors...
            "rgcn",  # too slow
        }:
            logger.warning(f"Skipping {path} due to explicit model ignore rule")
            continue

        output_path = output_root.joinpath(device.type, model, dataset, "_".join((reference, *remainder)))
        if output_path.exists():
            logger.debug(f"Skipping configuration {path} since output path exists {output_path}")
            continue

        # load configuration
        configuration = load_configuration(path)
        # reduce number of training epochs
        configuration["pipeline"]["training_kwargs"]["num_epochs"] = num_epochs
        # discard results
        configuration.pop("results", None)

        logger.info(f"Running configuration from {path}")
        logger.debug(pprint.pformat(configuration, indent=2, sort_dicts=True))
        try:
            result = pipeline_from_config(config=configuration)
        except TypeError as error:
            logger.error("Could not run pipeline", exc_info=error)
            continue

        # save results
        result.save_to_directory(
            directory=output_path,
            save_replicates=False,
            save_training=False,
        )

    data = []
    for i, path in enumerate(configuration_paths, start=1):
        reference, model, dataset, *remainder = path.stem.split("_")
        output_path = output_root.joinpath(
            device.type, model, dataset, "_".join((reference, *remainder)), "results.json"
        )
        if not output_path.exists():
            logger.warning(f"{output_path} is not existing")
            continue
        results = json.loads(output_path.read_text())
        times = results["times"]
        times["training"] /= num_epochs
        times["path"] = path.relative_to(configuration_root).as_posix()
        data.append(times)
    df = pandas.DataFrame.from_records(data=data)
    df = df.sort_values(by="path")[["path", "training", "evaluation"]]
    output_path = output_root.joinpath("summary.tsv")
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Written summary to {output_path}")

    # print
    print(f"{'path':50}  training  evaluation")
    for _, row in df.iterrows():
        print(f"{row['path']:50} {row['training']:8.2f}s   {row['evaluation']:8.2f}s")


if __name__ == "__main__":
    main()
