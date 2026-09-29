# Benchmarks

## CPU benchmarks with airspeed velocity

The [`benchmarks`](benchmarks) package contains benchmarks for
[airspeed velocity (asv)](https://asv.readthedocs.io), covering scoring,
training, evaluation, negative sampling, and triples factories. They use
synthetic graphs, so they do not require dataset downloads, and are sized to
run on a CPU.

Timings on shared machines, such as CI runners, are noisy. The benchmarks are
therefore meant for *relative* comparisons of two commits on the same machine,
rather than for absolute numbers.

### Check that the benchmarks run

Run each benchmark once in the current environment (also run in CI):

```console
$ tox -e benchmarks
```

### Compare two commits

From this directory, run

```console
$ pip install asv virtualenv
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
