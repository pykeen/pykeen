# Investigation: PyKEEN #1684 (ruff `SLF` rules) and the underlying design issues

## Context

PR #1684 enables the flake8-self (`SLF`) ruff rules. It ignores `SLF001` in `tests/**`, replaces
`inspect._empty` with `inspect.Parameter.empty`, and adds `# noqa: SLF001` to every remaining private access in
`src` (36 sites).

Only one of those changes fixes real code; the rest are ignores. The question investigated here is whether the private
accesses are symptoms of a sub-optimal split of responsibilities, so that fixing them improves the design and gives the
rule real value.

Line numbers refer to master at 5006a2bf. The claims were checked against the code by a separate review pass.

## Guiding principle

A leading underscore keeps *users* from relying on a member, so the internals can change freely. Making a member public
to satisfy the linter widens the supported API and defeats that purpose. Each site therefore gets one of three
treatments:

- **Public API:** the member is useful to users or is a real extension point. Making it public is a feature.
- **Private module function:** the logic is shared between PyKEEN classes but has no user value. Move it into a
  module-level function (e.g. in a `_`-prefixed module) and import it. Verified with ruff: `from _helpers import _f`
  followed by `_f()` is not flagged by `SLF001`, while `_helpers._f()` (attribute access on the module) is. This fixes the
  responsibility split without growing the public surface.
- **Keep `noqa: SLF001`:** the access is a legitimate cross-class or third-party internal and the refactor is not
  worth it.

Policy positions (choices, not code facts):

- Private members are not public API, so renames need no deprecated aliases.
- Downstream impact of renames: renaming an *abstract* method (`Representation._plain_forward`,
  `Model._get_entity_len`) breaks subclasses loudly (instantiation raises `TypeError`). Renaming a *non-abstract hook*
  (`Model._free_graph_and_cache`, `Stopper._write_from_summary_dict`) makes an existing override silently stop being
  called. Both warrant a changelog note.
- Because the principle above favours not growing the public API, the strength of the case for enabling `SLF` rests on
  the design cleanups (items 1, 2, 4, 6), not on renaming.

## Findings that point at a design problem

### 1. Stopper and checkpoint state: `Model._random_seed`, `Stopper._write_from_summary_dict`

- `TrainingLoop` assembles the checkpoint by reaching into `Model` (`self.model._random_seed`, `training_loop.py:1314`)
  and applies stopper state via a private method (`stopper._write_from_summary_dict`, `training_loop.py:448`).
- Export is already public: `Stopper.get_summary_dict()` (abstract, `stopper.py:41`, used at `training_loop.py:1288`).
  Only the apply step is private. `load_summary_dict_from_training_loop_checkpoint` is a static file loader.
- `_write_from_summary_dict` is a no-op in the base class, overridden in `early_stopping.py:311`, so it is a
  non-abstract hook (see the silent-override note above).
- Fix: make the apply step public (e.g. `load_summary_dict` / `set_summary_dict`), symmetric to `get_summary_dict`, and
  expose `Model.random_seed` as a read-only property. `models/meta/filtered.py:96` (`base._random_seed`) is covered too.
- Recommendation: **public API.** The stopper's state export is already public, so the apply step completes a symmetric
  interface, and `Model.random_seed` is plainly useful to users (reproducibility).

### 2. Callbacks driving the training loop: `_should_stop`, `_save_state`, `_train_epoch`, `_create_training_data_loader`

- `callbacks.py` sets `training_loop._should_stop = True` (377), calls `training_loop._save_state(...)` (382; its only
  outside caller), and `_validation_loss_amo_wrapper` calls `_train_epoch` (449) and `_create_training_data_loader` (451).
- Callbacks need a small public control surface on the loop:
  - `_should_stop = True` becomes `training_loop.request_stop()`.
  - the best-epoch save becomes a public `save_checkpoint(path, triples_factory)`.
  - `_validation_loss_amo_wrapper` delegates to `_train_epoch(..., backward=False)` under `@torch.inference_mode()` and
    `@maximize_memory_utilization(...)` with a per-loop hasher (`id(training_loop)`). It has a
    `# todo: create dataset only once`. A public `evaluate_loss(triples_factory, batch_size, slice_size)` on the loop
    would remove the callbacks' access to loop internals, but it would have to carry the memory-utilization wrapping
    and its caching key.
- Recommendation: **public API for `request_stop` and `save_checkpoint`**, since callbacks are an extension point and
  users write their own. `evaluate_loss` is borderline: it is only needed by one built-in callback, so a private module
  function taking the loop is the more conservative choice.

### 3. `Model._free_graph_and_cache`

- `TrainingLoop._free_graph_and_cache` (`training_loop.py:1257`) is a private wrapper called 11 times inside the loop
  (619, 1027, 1036, 1052, 1143, 1159, 1196, 1207, 1230, 1240, 1253). It calls the model method at line 1258 and also
  `torch.cuda.empty_cache()`. `callbacks.py:379` bypasses the wrapper.
- `Model._free_graph_and_cache` (`models/base.py:187`) is a no-op hook with no override anywhere in `src`, so the
  callback's call currently does nothing.
- Fix options: make the model method public (needed for the wrapper at 1258 anyway). Switching the callback to the loop's
  wrapper is a behaviour change (it would also empty the CUDA cache), not a pure refactor.
- Recommendation: **keep `noqa`** (or fold into item 2). The model method is a no-op hook with no user-facing purpose,
  so making it public adds API for nothing. If the early-stopping callback moves behind the loop's control surface
  (item 2), this access disappears without any new public member.

### 4. Inverse triples: `TriplesFactory._add_inverse_triples_if_necessary`

- Called from `bcwa.py:248`, `instances.py:270` and `instances.py:481`.
- The factory does *not* own the inverse flag: `triples_factory.py:298-300` pops `create_inverse_triples` from the
  factory state ("the inverse triples flag is owned by the model nowadays"), and `bcwa.py:248` takes it from
  `self.model.use_inverse_triples`. Passing the flag is therefore necessary.
- The real duplication is the expression
  `num_relations = 2 * tf.real_num_relations if create_inverse_triples else tf.real_num_relations`
  (`instances.py:274` and `:486`), which must stay consistent with the triples produced by the method.
- Fix: one factory method returning the triples together with the matching relation count, still taking the flag
  (e.g. `tf.get_training_triples(create_inverse_triples)`).
- Recommendation: **private module function.** Training-data construction is internal. A helper in the triples package
  that returns `(mapped_triples, num_relations)` removes both the `noqa` comments and the duplicated expression without
  extending `TriplesFactory`'s public surface.

### 5. `Model._prepare_batch` in `models/uncertainty.py`

- Called by `predict_hrt_uncertain` (`:215`), `predict_h_uncertain` (`:266`) and `predict_t_uncertain` (`:365`).
  `predict_r_uncertain` does not call it (no relation column to translate).
- `_prepare_batch` translates real relation IDs to internal ones and moves the batch to the device. This is a different
  concern than item 4 (appending inverse triples), though both deal with inverse-relation handling.
- Fix: either a public `Model` method, or `predict_uncertain_helper` accepts real IDs and translates itself. The latter
  needs an `index_relation` argument and special handling for `predict_r_uncertain`'s (h, t) batch.
- Recommendation: **private module function** (or move the translation into `predict_uncertain_helper`). The
  user-facing entry points are the `predict_*_uncertain` functions; the ID translation is an implementation detail.
  Note that `Model.predict_*` already uses `_prepare_batch` internally, so it can stay private on `Model` if the helper
  lives next to it.

### 6. `_process_batch_static` in `contrib/lightning.py` (`:182`, `:226`)

- The Lightning module calls `SLCWATrainingLoop._process_batch_static` (a `@classmethod`, `slcwa.py:128-129`, dispatching
  to `cls._process_grouped_batch_static`) and `LCWATrainingLoop._process_batch_static` (a `@staticmethod`,
  `lcwa.py:98-99`). The loops call these themselves too (`slcwa.py:284`, `lcwa.py:143`).
- This suggests the batch-to-loss logic belongs in a standalone function (or loss helper) shared by the training loop and
  Lightning, instead of a class-level method that Lightning has to reach into.
- Recommendation: **private module function** shared by the loops and Lightning. This is the clearest case where a
  split-of-responsibilities fix removes the access without any public API.

### 7. `Model._get_entity_len`

- Six outside call sites: `predict.py:1004`, `slcwa.py:191` and `:269`, `training_loop.py:131` and `:1097`,
  `filtered.py:160`. Overrides in `filtered`, `nbase`, `inductive`, `baseline` and `mocks`. `_get_num_targets` in
  `training_loop.py` wraps it.
- It is abstract-ish API for subclasses (renaming is loud for downstream models). A plain public `get_entity_len(mode)`
  is sufficient. Making the mode-dependent count an attribute of the mode or the triples factory would be cleaner but is
  a larger refactor.
- Recommendation: **keep `noqa`** or rename to a public `get_entity_len`. Renaming is loud for downstream models
  (abstract method) and adds to the API; the count is a reasonable thing to expose, but there is no user demand evident
  in the code. Low priority.

## Findings that were checked

### `Representation._plain_forward` (`representation.py`)

- Definitions: the abstract base at `:211` plus 14 overrides (10 more in `representation.py`, plus
  `message_passing.py`, `node_piece/representation.py`, `vision/representation.py`, `pyg.py`).
- Access sites: `:326` (`SubsetRepresentation`) and `:1380` (`CombinedRepresentation._combine`).
- `Representation.forward` applies unique-dedup, normalizer, regularizer and dropout around `_plain_forward`. The two
  access sites call the bases' `_plain_forward` directly, so any normalizer, regularizer, dropout or `unique` setting on
  a *base* representation is silently ignored there; only the wrapper's own post-processing applies. Calling `forward`
  instead would apply them twice where both base and wrapper configure them.
- The bypass is probably deliberate, but nothing in the code documents it; this is an inference. The wrapper's `unique`
  is handled by the inherited `Representation.forward` (defaulting to all bases' `unique`).
- Recommendation: **keep `noqa`.** It is an extension hook used between representation classes, and the underscore
  keeps users from calling it and bypassing normalization and dropout. Renaming it to a public name would invite exactly
  the misuse the underscore prevents. The rule then does not reach zero outside torch internals.

### `_head_indices` / `_tail_indices` in `nn/modules.py:2850-2851`

- The wrapper interaction copies the base's private fields. The public `head_indices` / `tail_indices` properties return
  the same information, except a `None` field becomes `range(len(entity_shape))`. That substitution is semantically
  identical for `head_shape` / `tail_shape`.
- Fix: `self._head_indices = base.head_indices`, `self._tail_indices = base.tail_indices`. No new API.
- Recommendation: **use the existing public properties.**
- Side finding (`modules.py:205-216`): `tail_indices` uses `range(len(self.tail_shape))` while `head_indices` uses
  `range(len(self.entity_shape))`. Same value today, but inconsistent; use `entity_shape` for symmetry.

### `ChildERModel._interaction` (`models/resolve.py:153`)

- Not read anywhere in `src`; the class does not use it (the instance is captured by the closure in `__init__`), and
  nothing in `nbase.py` or the other `src` models shadows the name.
- It *is* read by `tests/test_pipeline.py:121-122`
  (`assert isinstance(model_cls._interaction, TransEInteraction)`, `model_cls._interaction.p == 2`). It is therefore not
  dead code. Removing it requires changing that test (e.g. check the instantiated model's interaction instead), or
  keeping the attribute with a `noqa`.
- Recommendation: **change the test** to assert on an instantiated model's interaction, then delete the attribute and
  its `noqa`. The attribute exists only for the test, so it is not worth keeping.

## Smaller renames

- `ValueRange._coerce` (`metrics/utils.py:71`): static formatting helper used by `ValueRange.notate` (`:68`) and by
  `Metric.get_range` (`:139`, `:141`). Make it a module-level function.
- `Dataset._tup` (`datasets/base.py:412`): called 3 times on `self` (`:382`, `:392`, `:559`), twice in the module-level
  `dataset_similarity` (`:96`, `a._tup(), b._tup()`), and 4 times in tests (`tests/test_deteriorate.py:31`, `:49`).
  Making it a property (e.g. `factories`) changes the call syntax at all of these.
- Recommendation: **`ValueRange._coerce` as a private module function; `Dataset._tup` stays private with a private
  module function for `dataset_similarity`** (or keep the single `noqa`). Neither has user value, and `_tup` is used in
  tests, which are exempt from the rule.

## Not fixable: torch internals (5 sites)

Sparse `_nnz` / `_indices` / `_values` in `nn/utils.py` (3 sites) and `torch.nn.modules.batchnorm._BatchNorm` /
`torch.nn.modules.dropout._DropoutNd` in `utils.py` (2 sites). No public equivalents. These keep their
`noqa: SLF001`.

## Summary of recommendations

| Site | Treatment |
|---|---|
| `head_indices` / `tail_indices` in `nn/modules.py` | Use existing public properties (no new API) |
| `ChildERModel._interaction` | Change the test, delete the attribute |
| `Model.random_seed`, stopper state apply step | Public API |
| Loop control for callbacks (`request_stop`, `save_checkpoint`) | Public API |
| `evaluate_loss` for the validation-loss callback | Private module function |
| Inverse-triples training data | Private module function |
| `_prepare_batch` in `uncertainty.py` | Private module function |
| `_process_batch_static` for Lightning | Private module function |
| `ValueRange._coerce`, `Dataset._tup` in `dataset_similarity` | Private module function |
| `Model._free_graph_and_cache` | Keep `noqa` (disappears if item 2 is done) |
| `Model._get_entity_len` | Keep `noqa` (low priority) |
| `Representation._plain_forward` | Keep `noqa` |
| Torch internals (5 sites) | Keep `noqa` |

## Suggested sequencing

Self-contained, no new API, could go into #1684 right away (removes 2 `noqa`):

- `modules.py:2850-2851` use the public `head_indices` / `tail_indices` (and fix the `tail_indices` asymmetry).

Design PRs, one concern each:

1. Stopper and checkpoint state (item 1), including `Model.random_seed`. Adds public API.
2. Training loop control surface for callbacks (items 2 and 3; note the `empty_cache` behaviour change). Adds public API.
3. Private module functions for inverse-triples data (item 4), ID translation (item 5) and batch-to-loss shared with
   Lightning (item 6). No public API change.
4. Small cleanups: `ChildERModel._interaction` (with its test), `ValueRange._coerce`, `Dataset._tup`.

After these land, #1684 reduces to the ruff config plus about 10 remaining `noqa` comments: 5 torch internals,
2 `_plain_forward`, and the `_get_entity_len` and `_free_graph_and_cache` sites if left as is. That is much smaller than
the current 36, but it is not zero, so whether the rule is worth enabling at that point remains a maintainer judgment.
