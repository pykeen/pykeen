import pathlib
import pprint
import click
import logging
import more_click
from pykeen.constants import PYKEEN_BENCHMARKS
from pykeen.pipeline.api import pipeline_from_config, replicate_pipeline_from_config
from pykeen.utils import CONFIGURATION_FILE_FORMATS, get_devices, load_configuration, resolve_device
from pykeen.experiments.cli import HERE
from pykeen.version import get_git_hash

logger = logging.getLogger(__name__)


@click.command()
@click.option("-c", "--configuration-root", type=pathlib.Path, default=HERE)
@click.option("-o", "--output-root", type=pathlib.Path, default=PYKEEN_BENCHMARKS.joinpath("pipeline"))
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
    logging.info(f"Running on device: {device}")

    configuration_paths = sorted(
        path for ext in CONFIGURATION_FILE_FORMATS for path in configuration_root.rglob(f"*{ext}")
    )
    logger.info(f"Found {len(configuration_paths)} configurations under {configuration_root}")

    git_hash = get_git_hash()
    if git_hash == "UNHASHED":
        logger.warning("Could not parse git hash!")

    for path in configuration_paths:
        reference, model, dataset, *remainder = path.stem.split("_")
        output_path = output_root.joinpath(device.type, git_hash, model, dataset, "_".join((reference, *remainder)))
        if output_path.exists():
            logger.debug(f"Skipping configuration {path} since output path exists {output_path}")
            continue

        # load configuration
        configuration = load_configuration(path)
        # reduce number of training epochs
        configuration["pipeline"]["training_kwargs"]["num_epochs"] = num_epochs
        # discard results
        configuration.pop("results")

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


if __name__ == "__main__":
    main()
