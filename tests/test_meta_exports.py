"""Meta-tests for the public exports (``__all__``) of all PyKEEN modules."""

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path
from types import ModuleType

import pytest

import pykeen

# Public names that are defined in a module but missing from its ``__all__`` (or from the module not having an
# ``__all__`` at all). Sphinx only documents what is listed in ``__all__``, so these are undocumented or
# inconsistently exposed. The test fails if a new name is missing from ``__all__`` and if a name listed here is
# exported (or removed) by now, so fix modules by exporting the names and deleting the corresponding entry.
UNEXPORTED_ALLOWLIST: dict[str, set[str]] = {
    "pykeen.ablation.ablation": {"SplitToPathDict", "iter_unique_ids", "path_to_str"},
    "pykeen.checkpoints.base": {"ModelState", "get_model_state", "save_state_torch"},
    "pykeen.checkpoints.inspection": {"final_checkpoints", "simulate_checkpoints"},
    "pykeen.cli": {
        "HERE",
        "METRICS_SKIP",
        "METRIC_NAMES",
        "datasets",
        "evaluators",
        "get_metric_list",
        "get_readme",
        "hpo_samplers",
        "importers",
        "losses",
        "lr_schedulers",
        "ls",
        "main",
        "metrics",
        "models",
        "negative_samplers",
        "optimizers",
        "readme",
        "regularizers",
        "stoppers",
        "trackers",
        "train",
        "training_loops",
        "version",
    },
    "pykeen.constants": {
        "COLUMN_LABELS",
        "DEFAULT_DROPOUT_HPO_RANGE",
        "DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE",
        "PYKEEN_DEFAULT_CHECKPOINT",
        "PYKEEN_MODULE",
        "TARGET_TO_INDEX",
        "TARGET_TO_KEYS",
        "TARGET_TO_KEY_LABELS",
        "USER_DEFINED_CODE",
        "get_target_column",
    },
    "pykeen.contrib.lightning": {"lit_pipeline"},
    "pykeen.datasets.analysis": {"SUBSET_COLUMN_NAME", "get_relation_injectivity_df"},
    "pykeen.datasets.aristo": {"AristoV4"},
    "pykeen.datasets.base": {"DEFAULT_RATIOS"},
    "pykeen.datasets.biokg": {"URL"},
    "pykeen.datasets.ckg": {"CKGSimpleSource", "COLUMNS", "URL"},
    "pykeen.datasets.cli": {"IMG_DIR", "ROOT", "analyze", "degree", "expected_metrics", "main", "summarize", "verify"},
    "pykeen.datasets.codex": {
        "BASE_URL",
        "CoDExLarge",
        "CoDExMedium",
        "CoDExSmall",
        "LARGE_TEST_URL",
        "LARGE_TRAIN_URL",
        "LARGE_VALID_URL",
        "MEDIUM_TEST_URL",
        "MEDIUM_TRAIN_URL",
        "MEDIUM_VALID_URL",
        "SMALL_TEST_URL",
        "SMALL_TRAIN_URL",
        "SMALL_VALID_URL",
    },
    "pykeen.datasets.conceptnet": {"ConceptNet", "URL"},
    "pykeen.datasets.countries": {"BASE_URL"},
    "pykeen.datasets.cskg": {"URL"},
    "pykeen.datasets.db100k": {"BASE_URL"},
    "pykeen.datasets.dbpedia": {"BASE", "TEST_URL", "TRAIN_URL", "VALID_URL"},
    "pykeen.datasets.drkg": {"URL"},
    "pykeen.datasets.ea.combination": {
        "cat_shift_triples",
        "filter_map_alignment",
        "iter_entity_mappings",
        "merge_label_to_id_mapping",
        "merge_label_to_id_mappings",
        "swap_index_triples",
    },
    "pykeen.datasets.ea.openea": {
        "D_W",
        "D_Y",
        "EN_DE",
        "EN_FR",
        "GRAPH_PAIRS",
        "GRAPH_SIZES",
        "GRAPH_VERSIONS",
        "OPEN_EA_MODULE",
        "SIZE_100K",
        "SIZE_15K",
        "V1",
        "V2",
    },
    "pykeen.datasets.ea.wk3l": {"EA_SIDES_R", "EN_DE", "EN_FR", "GOOGLE_DRIVE_ID", "GRAPH_PAIRS", "WK3L_MODULE"},
    "pykeen.datasets.generate": {"LABELS", "main"},
    "pykeen.datasets.globi": {"URL"},
    "pykeen.datasets.hetionet": {"URL"},
    "pykeen.datasets.inductive.ilp_teru": {
        "BASE_URL",
        "FB_INDUCTIVE_INFERENCE_URL",
        "FB_INDUCTIVE_TEST_URL",
        "FB_INDUCTIVE_VALIDATION_URL",
        "FB_TRAIN_URL",
        "NELL_INDUCTIVE_INFERENCE_URL",
        "NELL_INDUCTIVE_TEST_URL",
        "NELL_INDUCTIVE_VALIDATION_URL",
        "NELL_TRAIN_URL",
        "WN_INDUCTIVE_INFERENCE_URL",
        "WN_INDUCTIVE_TEST_URL",
        "WN_INDUCTIVE_VALIDATION_URL",
        "WN_TRAIN_URL",
    },
    "pykeen.datasets.inductive.ilpc2022": {
        "BASE_URL",
        "LARGE_INFERENCE_TEST_URL",
        "LARGE_INFERENCE_URL",
        "LARGE_INFERENCE_VAL_URL",
        "LARGE_TRAIN_URL",
        "SMALL_INFERENCE_TEST_URL",
        "SMALL_INFERENCE_URL",
        "SMALL_INFERENCE_VAL_URL",
        "SMALL_TRAIN_URL",
    },
    "pykeen.datasets.kinships": {"HERE"},
    "pykeen.datasets.nations": {"HERE"},
    "pykeen.datasets.ogb": {
        "BioKGEvalDict",
        "BioKGTrainDict",
        "LOGGER",
        "NODE_TYPES",
        "WikiKG2EvalDict",
        "WikiKG2TrainDict",
        "load_partial_entity_mapping",
    },
    "pykeen.datasets.openbiolink": {"HQ_URL", "LQ_URL"},
    "pykeen.datasets.pharmebinet": {"RAW_URL"},
    "pykeen.datasets.pharmkg": {"BASE_URL", "RAW_URL", "TEST_URL", "TRAIN_URL", "VALID_URL"},
    "pykeen.datasets.primekg": {"URL"},
    "pykeen.datasets.umls": {"HERE"},
    "pykeen.datasets.utils": {"get_dataset", "iter_dataset_classes", "iter_dataset_instances"},
    "pykeen.datasets.wd50k": {"BASE_URL", "TRIPLES_TEST_URL", "TRIPLES_TRAIN_URL", "TRIPLES_VALID_URL", "WD50KT"},
    "pykeen.datasets.wikidata5m": {"TRANSDUCTIVE_URL"},
    "pykeen.evaluation.classification_evaluator": {"ClassificationMetricKey", "ScorePack"},
    "pykeen.evaluation.evaluation_loop": {"FilterIndex"},
    "pykeen.evaluation.evaluator": {
        "create_dense_positive_mask_",
        "create_sparse_positive_filter_",
        "evaluate",
        "get_candidate_set_size",
    },
    "pykeen.evaluation.rank_based_evaluator": {
        "HITS_PATTERN",
        "METRIC_PATTERN",
        "RANDOM_TARGET_SLICE",
        "RANKING_METRICS",
        "RankBasedMetricKey",
        "RankPack",
        "summarize_values",
    },
    "pykeen.experiments.cli": {"HERE", "ablation", "cmd_list", "optimize", "reproduce", "run", "validate"},
    "pykeen.experiments.inverse_stability": {"INVERSE_STABILITY", "main", "run_inverse_stability_workflow"},
    "pykeen.experiments.validate": {"get_configuration_errors", "iterate_config_paths"},
    "pykeen.hpo.cli": {"optimize"},
    "pykeen.hpo.hpo": {
        "ExtraKeysError",
        "Objective",
        "STOPPED_EPOCH_KEY",
        "suggest_discrete_power_int",
        "suggest_kwargs",
    },
    "pykeen.losses": {
        "DEFAULT_HPO_STRATEGY_POS_WEIGHT",
        "DEFAULT_HPO_STRATEGY_REDUCTION",
        "DEFAULT_MARGIN_HPO_STRATEGY",
        "NoSampleWeightSupportError",
        "UnsupportedLabelSmoothingError",
        "apply_label_smoothing",
        "prepare_negative_scores_for_softmax",
        "weighted_reduction",
    },
    "pykeen.metrics.classification": {"safe_divide"},
    "pykeen.metrics.ranking": {"MedianAbsoluteDeviation"},
    "pykeen.models.baseline.utils": {"sparsify"},
    "pykeen.models.cli.options": {"CLI_OPTIONS"},
    "pykeen.models.inductive.base": {"InductiveERModel"},
    "pykeen.models.nbase": {"iter_slices"},
    "pykeen.models.resolve": {"DimensionError"},
    "pykeen.models.unimodal.auto_sf": {"YAGO310_COEFFICIENTS"},
    "pykeen.nn.init": {"init_quaternions"},
    "pykeen.nn.modules": {"ConvEResolvedImageShape", "parallel_slice_batches", "parallel_unsqueeze"},
    "pykeen.nn.node_piece.anchor_search": {"DISTANCE_DTYPE"},
    "pykeen.nn.node_piece.cli": {"tokenize"},
    "pykeen.nn.pyg": {"FLOW_DIRECTIONS"},
    "pykeen.nn.representation": {
        "InvalidBaseIdsError",
        "Partition",
        "build_representation",
        "process_max_id",
        "process_shape",
    },
    "pykeen.nn.text.cache": {"PYOBO_PREFIXES_WARNED"},
    "pykeen.nn.utils": {"BaseShapeError", "iter_matrix_power"},
    "pykeen.nn.vision.cache": {"WIKIDATA_IMAGE_RELATIONS"},
    "pykeen.nn.weighting": {"softmax"},
    "pykeen.pipeline.api": {"compare_results", "save_pipeline_results_to_directory", "triple_hash"},
    "pykeen.pipeline.plot_utils": {"REDUCER_RELATION_WHITELIST"},
    "pykeen.regularizers": {"DEFAULT_REGULARIZER_WEIGHT_HPO_RANGE"},
    "pykeen.sampling.pseudo_type": {"create_index"},
    "pykeen.training.bcwa": {"BatchCWACollator", "BatchCWADataset"},
    "pykeen.training.callbacks": {
        "EvaluationLossTrainingCallback",
        "LearningRateSchedulerTrainingCallback",
        "OptimizerTrainingCallback",
    },
    "pykeen.training.training_loop": {"CHECKSUM_VERSION", "CheckpointMismatchError", "NoTrainingBatchError"},
    "pykeen.triples.analysis": {
        "CARDINALITY_TYPE_COLUMN_NAME",
        "CARDINALITY_TYPE_MANY_TO_MANY",
        "CARDINALITY_TYPE_MANY_TO_ONE",
        "CARDINALITY_TYPE_ONE_TO_MANY",
        "CARDINALITY_TYPE_ONE_TO_ONE",
        "CONFIDENCE_COLUMN_NAME",
        "COUNT_COLUMN_NAME",
        "ENTITY_ID_COLUMN_NAME",
        "ENTITY_LABEL_COLUMN_NAME",
        "ENTITY_POSITION_COLUMN_NAME",
        "FUNCTIONALITY_COLUMN_NAME",
        "HeadTailPairIndex",
        "INVERSE_FUNCTIONALITY_COLUMN_NAME",
        "PATTERN_TYPE_ANTI_SYMMETRY",
        "PATTERN_TYPE_COLUMN_NAME",
        "PATTERN_TYPE_COMPOSITION",
        "PATTERN_TYPE_IMPLICATION",
        "PATTERN_TYPE_INVERSION",
        "PATTERN_TYPE_SYMMETRY",
        "PatternMatch",
        "RELATION_CARDINALITY_TYPES",
        "RELATION_ID_COLUMN_NAME",
        "RELATION_LABEL_COLUMN_NAME",
        "RELATION_PATTERN_TYPES",
        "SUPPORT_COLUMN_NAME",
        "iter_binary_patterns",
        "iter_patterns",
        "iter_relation_cardinality_types",
        "iter_ternary_patterns",
        "iter_unary_patterns",
        "skyline",
        "triple_set_hash",
    },
    "pykeen.triples.leakage": {
        "get_candidate_pairs",
        "jaccard_similarity_scipy",
        "mapped_triples_to_sparse_matrices",
        "triples_factory_to_sparse_matrices",
    },
    "pykeen.triples.splitting": {"construct_uniform_probability", "split_fully_inductive", "split_semi_inductive"},
    "pykeen.triples.triples_factory": {
        "Condenser",
        "InvalidMappedTriplesShapeError",
        "Labeling",
        "TripleCondenser",
        "restrict_triples",
        "valid_triple_id_range",
    },
    "pykeen.triples.triples_numeric_literals_factory": {"create_matrix_of_literals"},
    "pykeen.triples.utils": {
        "EXTENSION_IMPORTER_RESOLVER",
        "InvalidRemappingLengthError",
        "PREFIX_IMPORTER_RESOLVER",
        "TRIPLES_DF_COLUMNS",
    },
    "pykeen.typing": {
        "RANK_OPTIMISTIC",
        "RANK_PESSIMISTIC",
        "RANK_REALISTIC",
        "RANK_TYPES",
        "RANK_TYPE_SYNONYMS",
        "SIDES",
        "SIDE_BOTH",
        "SIDE_MAPPING",
    },
    "pykeen.utils": {
        "AmbiguousDeviceError",
        "CONFIGURATION_FILE_FORMATS",
        "DeviceResolutionError",
        "add_cudnn_error_hint",
        "camel_to_snake",
        "circular_correlation",
        "determine_maximum_batch_size",
        "find",
        "getattr_or_docdata",
        "is_cudnn_error",
        "iter_weisfeiler_lehman",
        "load_configuration",
        "make_ones_like",
    },
    "pykeen.version": {"env_html", "env_table"},
}


_MODULE_INFOS = list(pkgutil.walk_packages(pykeen.__path__, prefix="pykeen."))
MODULE_NAMES = sorted(info.name for info in _MODULE_INFOS)
PACKAGE_NAMES = sorted(info.name for info in _MODULE_INFOS if info.ispkg)

# calls creating type variables, whose (upper-case) names are not public constants
_TYPE_VARIABLE_FACTORIES = {"TypeVar", "ParamSpec", "TypeVarTuple"}


def _is_type_variable(node: ast.Assign | ast.AnnAssign) -> bool:
    """Check whether an assignment creates a type variable."""
    value = node.value
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    return (isinstance(func, ast.Name) and func.id in _TYPE_VARIABLE_FACTORIES) or (
        isinstance(func, ast.Attribute) and func.attr in _TYPE_VARIABLE_FACTORIES
    )


def _import(name: str) -> ModuleType:
    """Import a module, skipping the test if an optional third-party dependency is not installed."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name is not None and error.name.split(".")[0] != "pykeen":
            pytest.skip(f"Missing optional dependency {error.name!r}")
        raise


def _defined_public_names(module: ModuleType) -> set[str]:
    """Get the public classes, functions and upper-case constants defined at the top level of a module."""
    source_file = inspect.getsourcefile(module)
    assert source_file is not None
    tree = ast.parse(Path(source_file).read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign | ast.AnnAssign) and not _is_type_variable(node):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(target.id for target in targets if isinstance(target, ast.Name) and target.id.isupper())
    return {name for name in names if not name.startswith("_")}


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_all_entries_exist(module_name: str) -> None:
    """Test that all names in ``__all__`` exist in the module, and are listed once."""
    module = _import(module_name)
    exported = getattr(module, "__all__", [])
    missing = sorted(name for name in exported if not hasattr(module, name))
    assert not missing, f"{module_name}.__all__ lists names that do not exist: {missing}"
    duplicates = sorted({name for name in exported if exported.count(name) > 1})
    assert not duplicates, f"{module_name}.__all__ lists names more than once: {duplicates}"


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_public_names_are_exported(module_name: str) -> None:
    """Test that public names defined in a module are exported via ``__all__``, apart from known omissions."""
    module = _import(module_name)
    unexported = _defined_public_names(module).difference(getattr(module, "__all__", []))
    allowed = UNEXPORTED_ALLOWLIST.get(module_name, set())
    new = sorted(unexported - allowed)
    assert not new, f"{module_name} defines public names which are not in __all__: {new}"
    stale = sorted(allowed - unexported)
    assert not stale, f"Remove these names from UNEXPORTED_ALLOWLIST['{module_name}'], they are exported now: {stale}"


def test_allowlist_modules_exist() -> None:
    """Test that the allowlist only refers to existing modules."""
    unknown = sorted(set(UNEXPORTED_ALLOWLIST).difference(MODULE_NAMES))
    assert not unknown, f"UNEXPORTED_ALLOWLIST refers to unknown modules: {unknown}"


@pytest.mark.parametrize("package_name", PACKAGE_NAMES)
def test_package_reexports_are_consistent(package_name: str) -> None:
    """Test that a name which a package exports is the same object as in the submodule which exports it."""
    package = _import(package_name)
    exported = set(getattr(package, "__all__", []))
    inconsistent = []
    for info in pkgutil.iter_modules(package.__path__, prefix=f"{package_name}."):
        try:
            submodule = _import(info.name)
        except pytest.skip.Exception:
            continue
        inconsistent.extend(
            f"{name} (differs from {info.name})"
            for name in exported.intersection(getattr(submodule, "__all__", []))
            if getattr(package, name, None) is not getattr(submodule, name, None)
        )
    assert not inconsistent, f"{package_name} exports names that differ from those of its submodules: {inconsistent}"
