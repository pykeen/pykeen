# Benchmarks

## CPU benchmarks with airspeed velocity

The [`benchmarks`](benchmarks) package contains benchmarks for
[airspeed velocity (asv)](https://asv.readthedocs.io), covering scoring,
training, evaluation, negative sampling, and triples factories. They are sized
to run on a CPU, and do not require dataset downloads:

- Benchmarks whose cost depends on the graph structure (training, negative
  sampling, triples factories) use Kinships, the largest dataset shipped with
  PyKEEN, since synthetic graphs hardly reproduce the patterns of real graphs,
  e.g., the number of triples per (head, relation) pair.
- Benchmarks whose cost mainly grows with the number of entities (scoring
  against all entities, evaluation) use a larger synthetic graph, since
  Kinships only has 104 entities, which would hide such costs.
- Evaluation additionally uses UMLS, since the cost of filtering depends on
  the number of known answers per query. In the synthetic graph, almost all
  queries have a single known answer, whereas UMLS has a skewed distribution,
  as real graphs do.

The configuration is in [`asv.conf.jsonc`](asv.conf.jsonc).

### Intended use and limitations

The benchmarks are meant to detect *regressions* by comparing two commits, not
as targets for optimization:

- Timings on shared machines, such as CI runners, are noisy. Only compare
  two commits run on the same machine, rather than absolute numbers.
- Small datasets exaggerate fixed costs, such as Python overhead and setup,
  and hide costs which grow with the size of the data. A change that looks
  small here may be large on a real dataset, and vice versa.
- The benchmarks run on a CPU, whereas PyKEEN models are typically trained and
  evaluated on a GPU, where costs are dominated by kernel launches, memory
  transfers, and memory usage instead. Speed-ups on a CPU need not carry over,
  and can even be slowdowns on a GPU.

Hence, justify optimizations with realistic workloads, i.e., real datasets, a
GPU where relevant, and profiling. Code that runs on the CPU in any case, such
as the computation of metrics with numpy, is well represented, though.

### Check that the benchmarks run

Run each benchmark once in the current environment (also run in CI):

```console
$ tox -e benchmarks
```

### Compare two commits

From this directory, run

```console
$ pip install "asv>=0.6.5" virtualenv
$ asv machine --yes
$ asv continuous --factor 1.2 master HEAD
```

This builds both commits in separate virtual environments, runs the benchmarks
for each, and reports the benchmarks whose timing changed by more than 20%.
The `Benchmarks` GitHub workflow runs this comparison when triggered manually,
and weekly for `master` against its state one week earlier.

To run a subset of the benchmarks, pass a regular expression, e.g.,
`-b training`.

## Other benchmarks

[`benchmark_splitting.py`](benchmark_splitting.py) is a standalone script which
benchmarks splitting the built-in datasets.
