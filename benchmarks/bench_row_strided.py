"""Model-free benchmark of vLLM's native FlashKDA call, including QKV packing."""

import argparse
import datetime
import importlib
import json
import statistics
from pathlib import Path

import torch


def capture(fn, calls):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    return graph


def measure(graph, calls, repeats):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    graph.replay()
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (calls * repeats)


def run_case(T, H, N, args):
    assert T % N == 0
    torch.manual_seed(42)
    D = 128
    packed = torch.randn(1, T, 3, H, D, dtype=torch.bfloat16, device="cuda")
    qkv = packed.unbind(2)
    dense = tuple(x.contiguous() for x in qkv)
    g = torch.randn(1, T, H, D, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn(1, T, 2, H, dtype=torch.bfloat16, device="cuda")[:, :, 0]
    A_log, dt_bias = torch.zeros(H, device="cuda"), torch.zeros(H, D, device="cuda")
    op = getattr(torch.ops, args.namespace)
    initial = torch.randn(N, H, D, D, device="cuda") * 0.01
    final = torch.empty_like(initial)
    out = torch.empty(1, T, H, D, device="cuda", dtype=torch.bfloat16)
    cu = torch.arange(N + 1, device="cuda", dtype=torch.int32) * (T // N)
    workspace = torch.empty(op.get_workspace_size(T, H, N), device="cuda", dtype=torch.uint8)

    def call(inputs):
        op.fwd(
            *inputs, g.contiguous(), beta, D ** -0.5, out, workspace, A_log, dt_bias, -5.0,
            initial.contiguous(), final, cu.contiguous(), None, None,
        )

    arms = {"copies_plus_flashkda": lambda: call(tuple(x.contiguous() for x in qkv)),
            "dense_flashkda": lambda: call(dense)}
    if args.strided:
        call(dense)
        expected = (out.clone(), final.clone())
        call(qkv)
        for a, b in zip((out, final), expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        arms["strided_flashkda"] = lambda: call(qkv)
    graphs = {name: capture(fn, args.graph_calls) for name, fn in arms.items()}
    raw = {name: [] for name in arms}
    for round_id in range(args.rounds):
        names = list(arms)
        if round_id % 2:
            names.reverse()
        for name in names:
            raw[name].append(measure(graphs[name], args.graph_calls, args.repeats))
    medians = {name: statistics.median(values) for name, values in raw.items()}
    result = {"tokens": T, "heads": H, "head_dim": D, "sequences": N,
              "tokens_per_seq": T // N,
              "qkv_strides": [list(x.stride()) for x in qkv],
              "state_dtype": str(initial.dtype), "cu_dtype": str(cu.dtype),
              "median_us": medians, "rounds_us": raw}
    if args.strided:
        a, b = medians["copies_plus_flashkda"], medians["strided_flashkda"]
        result.update(speedup=a / b, latency_reduction_pct=(a - b) / a * 100,
                      consumer_change_pct=(b / medians["dense_flashkda"] - 1) * 100)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path)
    parser.add_argument("--namespace", default="flash_kda", choices=["flash_kda", "_flashkda_C"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strided", action="store_true")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--graph-calls", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--case", action="append", help="tokens:heads:sequences")
    args = parser.parse_args()
    if args.library:
        torch.ops.load_library(str(args.library))
    else:
        importlib.import_module("flash_kda" if args.namespace == "flash_kda" else "vllm._flashkda_C")
    cases = args.case or [f"{T}:{H}:{T // 2048}" for T in (4096, 8192, 16384)
                          for H in (16, 32, 64)] + ["128:32:1"]
    data = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "namespace": args.namespace, "device": torch.cuda.get_device_name(),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "mode": "CUDA graphs", "rounds": args.rounds,
            "graph_calls": args.graph_calls, "repeats": args.repeats,
            "results": []}
    for case in cases:
        data["results"].append(run_case(*map(int, case.split(":")), args))
        if args.output:
            args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
