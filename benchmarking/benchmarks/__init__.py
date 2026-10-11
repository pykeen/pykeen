"""CPU benchmarks for PyKEEN, run with airspeed velocity (asv).

All benchmarks run on a CPU, without dataset downloads. See ``benchmarking/README.md``.
"""

import torch

# use a single thread for more stable timings, which depend less on the number of cores and on other load of the
# machine. asv runs each benchmark in a separate process, which imports this package first.
torch.set_num_threads(1)
