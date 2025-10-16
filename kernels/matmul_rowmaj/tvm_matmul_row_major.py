import argparse
import tvm
from tvm import te, autotvm

FUNC_NAME = "matmul"
log_file = "matmul_te_1tile.log"

# CLI parameters

parser = argparse.ArgumentParser(description="Tuning matmul TE 1-tile with TVM")
parser.add_argument("--M", type=int, default=1024, help="Size M")
parser.add_argument("--N", type=int, default=1024, help="Size N")
parser.add_argument("--K", type=int, default=1024, help="Size K")
parser.add_argument("--trials", type=int, default=None, help="Num of trials")
parser.add_argument("--dtype", type=str, default="float32", help="Type of tensors")
parser.add_argument("--symbol", type=str, default="matmul", help="Function name")
parser.add_argument("--cpu", type=str, default="cascadelake", help="Target CPU")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()

M, N, K, dtype = args.M, args.N, args.K, args.dtype
cpu = args.cpu
target = tvm.target.Target(f"llvm -mcpu={cpu}")

#

if args.trials:

    def n_trial_heuristic(space_length):
        return args.trials
else:

    def n_trial_heuristic(space_length):
        return space_length // 2

# Template autoTVM


@autotvm.template("gemm.te.1tile")
def te_matmul_1tile(M, N, K, dtype):
    # Define input placeholders:
    # A is an (M x K) matrix, B is a (K x N) matrix.
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)

    # Reduction axis over the shared dimension K.
    k = te.reduce_axis((0, K), name="k")

    # Define the compute for matrix multiplication:
    # C[i, j] = sum over k of A[i, k] * B[k, j]
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    # Create a schedule for the compute and get an AutoTVM config handle.
    s = te.create_schedule(C.op)
    cfg = autotvm.get_config()

    # Extract the parallel and reduction axes of C.
    i, j = C.op.axis
    r = C.op.reduce_axis[0]

    # Declare tunable tiling strategies for i, j (parallel) and k (reduction).
    # Each split will produce outer and inner tiles (num_outputs=2).
    cfg.define_split("tile_i", i, num_outputs=2)
    cfg.define_split("tile_j", j, num_outputs=2)
    cfg.define_split("tile_k", r, num_outputs=2)

    # Apply the chosen tiling to each axis; AutoTVM will fill in the split/tiling factors.
    io, ii = cfg["tile_i"].apply(s, C, i)
    jo, ji = cfg["tile_j"].apply(s, C, j)
    ko, ki = cfg["tile_k"].apply(s, C, r)

    # Set the final loop order: iterate outer tiles first, then inner ones.
    s[C].reorder(io, jo, ko, ii, ki, ji)

    # Vectorize the innermost j tile for SIMD execution.
    s[C].vectorize(ji)

    # Unroll small inner loops to reduce loop overhead and expose ILP.
    s[C].unroll(ki)
    s[C].unroll(ii)

    return s, [A, B, C]


# Task

task = autotvm.task.create(
    "gemm.te.1tile",
    args=(M, N, K, dtype),
    target=target,
)

n_trial = n_trial_heuristic(len(task.config_space))

# Tuning

measure_opt = autotvm.measure_option(
    builder=autotvm.LocalBuilder(build_func="default"),
    runner=autotvm.LocalRunner(number=5, repeat=1, min_repeat_ms=200),
)

callbacks = [autotvm.callback.log_to_file(log_file)]
if args.verbose:
    callbacks += [
        autotvm.callback.progress_bar(n_trial),
    ]

tuner = autotvm.tuner.GridSearchTuner(task)
tuner.tune(n_trial=n_trial, measure_option=measure_opt, callbacks=callbacks)


# Pick the best config

with autotvm.apply_history_best(log_file):
    with tvm.target.Target(target):
        s, [A, B, C] = te_matmul_1tile(M, N, K, dtype)
        rt_mod = tvm.build(s, [A, B, C], target=target, name=f"{args.symbol}")

# Print assembly


def to_c_global_asm(asm_text: str, symbol: str) -> str:
    lines = asm_text.strip().splitlines()

    has_text = any(line.strip().startswith(".text") for line in lines)
    has_globl = any(f".globl {symbol}" in line for line in lines)
    has_type = any(line.strip().startswith(f".type {symbol},") for line in lines)

    out = []
    if not has_text:
        out.append(".text")
    if not has_globl:
        out.append(f".globl {symbol}")
    if not has_type:
        out.append(f".type {symbol},@function")

    out.extend(lines)

    quoted = "\n".join(
        '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '\\n"' for s in out
    )
    return "__asm__(\n" + quoted + "\n);\n"


asm = rt_mod.get_source("asm")
print(to_c_global_asm(asm, args.symbol))
