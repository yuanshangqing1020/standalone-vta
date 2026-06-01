# VTA Simulators

Chinese: [`README_cn.md`](README_cn.md)

This directory contains two **complementary** VTA simulators (not interchangeable duplicates):

| Subdirectory | Abbrev. | Language | Purpose |
|--------------|---------|----------|---------|
| [`functional_simulator/`](functional_simulator/) | **FSIM** | C++ | Functional correctness: run `instructions.bin`, get the right tensor result **fast** |
| [`cycle_accurate_simulator/`](cycle_accurate_simulator/) | **TSIM** | Chisel/Scala | Cycle-accurate hardware model: timing, pipelines, detailed tests **slower** |

## FSIM entry points (same C++ core)

The functional simulator ships **two executables** (old name `vta_simulator` is retired):

- **`build/fsim_single_layer`** — one VTA layer at a time, per-layer `input*.bin`  
- **`build/fsim_nn`** — full quantized network via `dependency.csv`  

Both link `sim_driver` / `sim_tlpp` / `virtual_memory`. Details: [`functional_simulator/README.md`](functional_simulator/README.md) ([`README_cn.md`](functional_simulator/README_cn.md)) and [`docs/fsim_nn与fsim_single_layer_cn.md`](../../docs/fsim_nn与fsim_single_layer_cn.md).

## TSIM

Uses sbt tests such as `cli.ComputeApp_lenet5_layer1`. Many flows read the same `compiler_output/` binaries as FSIM for single-layer smoke tests (`make test_gemm`). JSON-based tests under `src/test/resources/` are a separate TSIM path.

See [`cycle_accurate_simulator/README.md`](cycle_accurate_simulator/README.md).

## Not simulators (often confused)

- **`reference_onnx.py` / `check_bin.py`** — ONNX golden reference and bit-accurate check against `final_output.bin`  
- **Compilers** under `src/compiler/` — produce `compiler_output/`, they do not execute VTA instructions  

## Quick choice

| Goal | Use |
|------|-----|
| 16×16 GEMM smoke test | FSIM `fsim_single_layer` + TSIM `ComputeApp` (`make test_gemm`) |
| ONNX whole-network verify | FSIM `fsim_nn` + `check` (`make run`) |
| Cycle-level hardware debug | TSIM `ComputeApp_*` |
