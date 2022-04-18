import json
import logging
import pathlib
import pprint
from typing import Any, Mapping
import platform
import click
import more_click
import pandas
import torch

from pykeen.constants import PYKEEN_BENCHMARKS
from pykeen.experiments.cli import HERE
from pykeen.pipeline.api import pipeline_from_config
from pykeen.utils import CONFIGURATION_FILE_FORMATS, format_relative_comparison, load_configuration, resolve_device
from pykeen.version import get_git_hash

logger = logging.getLogger(__name__)


def _collect_system_information() -> Mapping[str, Any]:
    uname = platform.uname()
    result = dict(
        system=uname.system,
        release=uname.release,
        machine=uname.machine,
        torch_version=torch.__version__,
    )
    if torch.cuda.is_available():
        # e.g. _CudaDeviceProperties(name='Quadro RTX 8000', major=7, minor=5, total_memory=48601MB, multi_processor_count=72)
        properties = torch.cuda.get_device_properties(device=torch.device("cuda"))
        result["gpu"] = dict(
            name=properties.name,
            total_memory=properties.total_memory,
            compute_capability=(properties.major, properties.minor),
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
        )
    return result


@click.command()
@click.option("-c", "--configuration-root", type=pathlib.Path, default=HERE)
@click.option("-o", "--output-root", type=pathlib.Path, default=PYKEEN_BENCHMARKS.joinpath("pipeline", get_git_hash()))
@click.option("-e", "--num-epochs", type=int, default=5)
@click.option("--debug", is_flag=True)
@more_click.log_level_option()
def main(
    configuration_root: pathlib.Path,
    output_root: pathlib.Path,
    num_epochs: int,
    log_level: str,
    debug: bool,
):
    """"""
    logging.basicConfig(level=log_level)

    device = resolve_device(device=None)
    logging.info(f"Running on device: {device}")

    system_information = _collect_system_information()

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
        configuration = dict(load_configuration(path))
        # reduce number of training epochs
        configuration["pipeline"]["training_kwargs"]["num_epochs"] = num_epochs
        # discard results
        configuration.pop("results", None)
        # add system information to metadata
        configuration.setdefault("metadata", {})
        configuration["metadata"]["system"] = system_information

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
            save_metadata=True,
            save_replicates=False,
            save_training=False,
        )

        if debug:
            break

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
        data.append(
            {
                "training": results["times"]["training"] / num_epochs,
                "evaluation": results["times"]["evaluation"],
                "batch_size.evaluation": results["evaluator_metadata"]["batch_size"],
                "path": path.relative_to(configuration_root).as_posix(),
            }
        )
    df = pandas.DataFrame.from_records(data=data)
    df = df[sorted(df.columns)]
    df = df.sort_values(by="path")
    output_path = output_root.joinpath("summary.tsv")
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Written summary to {output_path}")

    # print
    print(f"{'path':50}  training  evaluation")
    for _, row in df.iterrows():
        print(f"{row['path']:50} {row['training']:8.2f}s   {row['evaluation']:8.2f}s")


if __name__ == "__main__":
    main()
