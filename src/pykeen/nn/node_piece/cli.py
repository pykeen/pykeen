"""Command-Line Interface for pre-computing tokenizations for NodePiece."""
import copy
import logging
import math
import pathlib
from typing import Optional

import click
import more_click

from .anchor_search import anchor_searcher_resolver
from .anchor_selection import anchor_selection_resolver
from .loader import TorchPrecomputedTokenizerLoader
from ...constants import PYKEEN_MODULE
from ...datasets import dataset_resolver
from ...datasets.utils import _digest_kwargs
from ...utils import flatten_dictionary, get_edge_index, load_configuration

logger = logging.getLogger(__name__)


@click.command()
@dataset_resolver.get_option("-d", "--dataset", as_string=True, required=True)
@click.option("-c", "--configuration-path", type=pathlib.Path, default=None)
@click.option("-k", "--num-tokens", type=int, default=None)
@click.option("-a", "--num-anchors", type=int, default=None)
@click.option("-o", "--output-path", type=pathlib.Path, default=None)
@more_click.log_level_option()
@click.option("--force", is_flag=True)
def tokenize(
    dataset: str,
    configuration_path: pathlib.Path,
    num_tokens: Optional[int],
    num_anchors: Optional[int],
    output_path: Optional[pathlib.Path],
    log_level: str,
    force: bool,
):
    """Pre-compute tokenization."""
    logging.basicConfig(level=log_level)

    logger.info(f"Loading dataset: {dataset}")
    dataset_instance = dataset_resolver.make(dataset)

    # heuristic
    if num_anchors is None:
        num_anchors = max(2, int(math.ceil(math.sqrt(dataset_instance.num_entities))))
        logger.info(f"Inferred number of anchors using sqrt(num_entities) heuristic: {num_anchors}")

    # heuristic
    if num_tokens is None:
        num_tokens = max(2, int(math.ceil(math.sqrt(num_anchors))))
        logger.info(f"Inferred number of tokens using sqrt(num_anchors) heuristic: {num_tokens}")

    if configuration_path is None:
        logger.warning("No configuration path provided. Will use defaults.")
        configuration = {}
    else:
        logger.info(f"Loading tokenization configuration from {configuration_path}")
        configuration = dict(load_configuration(configuration_path))

    if output_path is None:
        # calculate configuration digest
        _configuration = copy.deepcopy(configuration)
        _configuration["num_anchors"] = num_anchors
        _configuration["num_tokens"] = num_tokens
        digest = _digest_kwargs(flatten_dictionary(_configuration))
        output_path = PYKEEN_MODULE.join(__name__.replace(".cli", ""), dataset_resolver.normalize(dataset)).joinpath(
            f"{digest}.pt"
        )
        logger.info(f"Inferred output path: {output_path}")
    if output_path.exists():
        logger.warning(f"Output path exists: {output_path}")
        if not force:
            logger.info("Existing file will not be overwritten. To enforce this, pass `--force`")
            quit()

    # create anchor selection instance
    selection_config = configuration.pop("selection", {})
    selection = selection_config.pop("class", None)  # TODO: better key?
    selection_instance = anchor_selection_resolver.make(selection, pos_kwargs=selection_config, num_anchors=num_anchors)
    logger.info(f"Created anchor selection instance: {selection_instance}")

    # select anchors
    edge_index = get_edge_index(triples_factory=dataset_instance.training).numpy()
    anchors = selection_instance(edge_index=edge_index)
    logger.info(f"Selected {len(anchors)} anchors")

    # anchor search (=anchor assignment?)
    searcher_config = configuration.pop("search", {})
    searcher = searcher_config.pop("class", None)
    searcher_instance = anchor_searcher_resolver.make(searcher, pos_kwargs=searcher_config)
    logger.info(f"Created anchor searcher instance: {searcher_instance}")

    # assign anchors
    sorted_anchor_ids = searcher_instance(edge_index=edge_index, anchors=anchors, k=num_tokens)

    # save
    TorchPrecomputedTokenizerLoader.save(
        path=output_path,
        order=sorted_anchor_ids,
        anchor_ids=anchors,
    )

    logger.info(f"Saved tokenization to {output_path}")
