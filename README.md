# standalone-vta

A maintained, unified, and extended Versatile Tensor Accelerator (VTA) ecosystem.

## Overview

This repository addresses the limitations of the original VTA project by providing:

* **Unified simulation:** A consistent input format (raw binary files + CSV metadata) for both the functional simulator (C++, FSIM) and the cycle-accurate simulator (CHISEL, TSIM).
* **Extended cycle-accurate simulation:** Multiple test cases for Compute / Load / Store and other submodules.
* **Standalone compiler:** An open-source, TVM-independent compiler that generates VTA binaries from VTA IR (JSON) or tutorial scripts.

VTA targets tensor operations such as matrix multiplication in CNNs. This project focuses on a reproducible compile → simulate → verify workflow.

## Toolchain

The end-to-end pipeline has two compilation stages:

```
qONNX model  →  vta_backend.py (NN compiler, stage 1)  →  VTA IR (JSON)
                                                          ↓
VTA IR (JSON)  →  main_vta_compiler.py (VTA compiler, stage 2)  →  binaries + instructions  →  simulators
```

| Stage | Entry | Input | Output |
|-------|-------|-------|--------|
| **Stage 1 (NN compiler)** | `vta_backend.py` | Quantized ONNX (qONNX) | VTA IR JSON, `dependency.csv`, weight/bias bins |
| **Stage 2 (VTA compiler)** | `main_vta_compiler.py` | VTA IR JSON | Instruction stream, UOP, block-layout bins, DRAM address tables |

See the Chinese docs for detailed walkthroughs: [`docs/nn_compiler_cn.md`](docs/nn_compiler_cn.md), [`docs/main_vta_compiler_cn.md`](docs/main_vta_compiler_cn.md).

Hardware parameters are defined in a single config file: [`config/vta_config.json`](config/vta_config.json). See [`docs/vta_config_cn.md`](docs/vta_config_cn.md).

## Repository Structure

| Path | Description |
|------|-------------|
| `tutorials/` | Tutorial notebooks and scripts (`tutorial1_data_definition.py`, `tutorial2_operations_definition.py`) |
| `src/compiler/` | VTA compiler (`data_definition`, `operations_definition`, `nn_compiler`, etc.) |
| `src/simulators/functional_simulator/` | C++ **functional simulation** (executables `build/fsim_single_layer`, `build/fsim_nn`) |
| `src/simulators/cycle_accurate_simulator/` | CHISEL **cycle-accurate simulation** (sbt tests, e.g. `ComputeApp_lenet5_layer1`) |
| `examples/` | End-to-end examples (`make test_gemm`, `make compile_and_run`, etc.) |
| `compiler_output/` | Binaries and CSV written by the compiler / tutorials (default input for simulators) |
| `config/vta_config.json` | VTA hardware parameters |
| `docs/` | Detailed Chinese documentation (see [Documentation Index](#documentation-index) below) |
| `environment_setup/standalone-vta.yml` | Conda environment definition |

## Dependencies

### 1. Base (tutorials and functional simulation)

* **g++** (C++17), **make**, **python3**
* **Conda environment:**

```bash
conda env create -f environment_setup/standalone-vta.yml
conda activate standalone-vta
```

### 2. Cycle-accurate simulation (TSIM, optional)

* **JDK 17**
* **sbt** (Scala Build Tool)
* **Verilator >= 5.x** (CHISEL generates Verilog for simulation)

Full installation steps are in [Cycle-Accurate Simulator (TSIM) → TSIM Environment Setup](#tsim-environment-setup) below.

---

## Quick Start

### A. Official examples (recommended first run)

From the repository root `standalone-vta/`:

```bash
conda activate standalone-vta
cd examples
make help          # list all targets
make test_gemm     # 16×16 GEMM: VTA compile + FSIM + TSIM
```

On success you get `compiler_output/` (binaries and CSV) and `log_output/` (logs). Step-by-step details: [`docs/MAKE_TEST_GEMM_cn.md`](docs/MAKE_TEST_GEMM_cn.md).

### B. Tutorial scripts (LeNet-5 layer 1)

```bash
cd tutorials
python3 tutorial1_data_definition.py       # Tutorial 1: Im2Row, padding, tiling (console validation, no full bin output)
python3 tutorial2_operations_definition.py # Tutorial 2: UOP/insn demo + writes uop.bin, instructions.bin
```

The header of `tutorial2_operations_definition.py` contains **complete simulator execution steps** (TSIM / FSIM and `compiler_output` file reference).

### C. ONNX full-network example (NN compile + FSIM + numerical check)

```bash
cd examples
make gen_onnx_qconv                                    # optional: regenerate minimal single-layer model
make run ONNX_FILE=onnx/qlinearconv_debug.onnx         # full pipeline
```

Step-by-step details: [`docs/MAKE_ONNX_RUN_cn.md`](docs/MAKE_ONNX_RUN_cn.md). The default `ONNX_FILE=onnx/qyolo_pattern.onnx` is larger; use `qlinearconv_debug.onnx` first to verify your environment.

---

## Functional Simulator (FSIM)

Path: `src/simulators/functional_simulator/`

### Executables (note: not `vta_simulator`)

Older docs refer to `vta_simulator` and `functional_simulator`; those names have changed. The Makefile **actually builds**:

| Target | Executable | Purpose |
|--------|------------|---------|
| `make build/fsim_single_layer` | `build/fsim_single_layer` | **Single layer:** reads `compiler_output/`, runs one layer per `layers_name.csv` |
| `make build/fsim_nn` | `build/fsim_nn` | **Full network:** multi-layer chain and post-processing |
| `make execute` | (same as above) | **Build** `fsim_single_layer` if needed, then **run** it |
| `make nn_execute` | (same as above) | Build and run `fsim_nn` |

> **Note:** `make all` targets `build/functional_simulator`, but the Makefile has **no** link rule for that name. Use the explicit targets above instead.

### What does `make execute` do?

1. Check and build `build/fsim_single_layer` (includes `vta_config.py`, simulation core, `external_lib/tvm`, etc.).
2. Run `build/fsim_single_layer` from the `functional_simulator/` directory.
3. The program reads from **`../../../compiler_output/`** (relative to `functional_simulator/`, i.e. repo-root `compiler_output/`):
   * `layers_name.csv` — layer count and filename suffix per layer
   * `metadata{suffix}.csv` — matrix dimensions
   * `input{suffix}.bin`, `weight{suffix}.bin`, `accumulator{suffix}.bin`, `uop{suffix}.bin`, `instructions{suffix}.bin`, etc.
4. Calls `VTADeviceRun()` to execute the instruction stream and print results (default `doPrint = true`).

**Running only `tutorials/tutorial2_operations_definition.py`** produces `uop.bin` and `instructions.bin` only — not enough for FSIM alone. You also need data binaries and CSV (see section 6 of `tutorials/tutorial2_operations_definition.py`, or copy from `examples_compute/lenet5_layer1/`).

### FSIM single-layer example (with tutorial instructions)

From the repository root:

```bash
# 1) Generate instructions (if not done yet)
cd tutorials && python3 tutorial2_operations_definition.py && cd ..

# 2) Copy data + instructions from TSIM test resources to compiler_output
EX=src/simulators/cycle_accurate_simulator/src/test/resources/examples_compute/lenet5_layer1
cp "$EX"/input.bin "$EX"/weight.bin "$EX"/accumulator.bin "$EX"/out.bin \
   "$EX"/memory_addresses.csv compiler_output/
cp "$EX"/expected_out_sram.bin compiler_output/expected_out.bin
cp compiler_output/uop.bin compiler_output/instructions.bin   # skip if tutorial2_operations_definition.py already generated them

# 3) CSV required for single-layer FSIM (empty suffix = filenames without QLinearConv etc.)
printf 'nb_vta_ir,1,False\n0,,0\n' > compiler_output/layers_name.csv
printf 'Matrix (or Block Size),Nb rows,Nb columns,Is it square?\nBS,16,16,True\nA,784,32,True\nX,784,16,True\nY,0,0,True\nC,784,16,True\n' \
  > compiler_output/metadata.csv

# 4) Build and run
cd src/simulators/functional_simulator
make build/fsim_single_layer
./build/fsim_single_layer
# or: make execute
```

---

## Cycle-Accurate Simulator (TSIM)

Path: `src/simulators/cycle_accurate_simulator/`

Tests such as `ComputeTest.scala` use binaries under `src/test/resources/examples_compute/<case>/` directly — **no JSON required** (JSON is used by other tests under `src/test/scala/simulator/`).

### LeNet-5 layer 1 (aligned with `tutorial2_operations_definition.py` / `insn_lenet5_layer1.py`)

```bash
# from repository root
cd tutorials && python3 tutorial2_operations_definition.py && cd ..

cp compiler_output/uop.bin compiler_output/instructions.bin \
   src/simulators/cycle_accurate_simulator/src/test/resources/examples_compute/lenet5_layer1/

cd src/simulators/cycle_accurate_simulator
sbt "testOnly cli.ComputeApp_lenet5_layer1"
```

That resource directory already contains `input.bin`, `weight.bin`, `accumulator.bin`, `memory_addresses.csv`, `expected_out_sram.bin`, etc.; you typically only replace newly generated `uop.bin` / `instructions.bin`.

### TSIM environment setup

The cycle-accurate simulator is based on Scala / CHISEL. The first run downloads Maven dependencies and may take a while. The steps below use **Ubuntu / Debian (including WSL2)**; on other systems install equivalent JDK 17, sbt, and Verilator versions.

#### 1. Java (JDK 17)

```bash
sudo apt update && sudo apt install -y openjdk-17-jdk

# verify
java -version
# should show 17.x
```

#### 2. sbt (Scala Build Tool)

```bash
echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list
curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo apt-key add -
sudo apt update && sudo apt install -y sbt

# verify
sbt sbtVersion
```

**Slow Maven access:** If sbt times out on Maven Central, configure a mirror (e.g. Aliyun) in `~/.sbt/repositories` and set `SBT_OPTS="-Dsbt.override.build.repos=true"`.

#### 3. Verilator (>= 5.x)

CHISEL simulation passes generated Verilog to Verilator to build and run C++. Distro packages often ship Verilator 4.x, which is **too old** — install from source (about 10–20 minutes):

```bash
sudo apt install -y git perl python3 make autoconf g++ flex bison \
    libfl2 libfl-dev libgoogle-perftools-dev numactl perl-doc \
    libunwind-dev zlib1g-dev ccache help2man

cd ~
git clone https://github.com/verilator/verilator
cd verilator
git checkout v5.020
autoconf
./configure
make -j"$(nproc)"
sudo make install

# verify (should be >= 5.0)
verilator --version
```

#### 4. Verify TSIM compiles

```bash
cd standalone-vta/src/simulators/cycle_accurate_simulator
sbt compile
```

After a successful compile, run `sbt "testOnly cli.ComputeApp_lenet5_layer1"` or other `ComputeApp_*` tests as above.

**Note:** `make tsim` in `examples/Makefile` runs `sbt "runMain org.scalatest.run cli.ComputeApp"`, which is **not** the same entry as `testOnly cli.ComputeApp_lenet5_layer1`. For the tutorial LeNet-5 layer 1, use the latter.

---

## Common `compiler_output/` Files

| File | Source | FSIM single layer | TSIM `lenet5_layer1` |
|------|--------|-------------------|----------------------|
| `instructions.bin` | `operations_definition` / `tutorial2_operations_definition.py` | required | required |
| `uop.bin` | same | required | required |
| `input.bin` / `weight.bin` | `data_definition` / example resources | required | bundled in resource dir |
| `accumulator.bin` | same | required | bundled in resource dir |
| `layers_name.csv` | VTA compiler or manual | required | not needed (test embeds paths) |
| `metadata*.csv` | same | required | not needed |
| `inputQLinearConv*.bin` etc. | full-network ONNX compile in `examples` | multi-layer FSIM | — |

If the directory already has full-network ONNX artifacts (`QLinearConv*.bin`), they are unrelated to `tutorials/tutorial2_operations_definition.py`; that script only overwrites `uop.bin` and `instructions.bin` in the same directory.

---

## Instruction Decoding (debug)

To inspect instruction fields without running a simulator:

```bash
cd src/compiler/vta_compiler/operations_definition
python3 structures.py
```

Or in Python: `from structures import decode_vta_insn, decode_uop` (see [`docs/VTA_ISA_REFERENCE_cn.md`](docs/VTA_ISA_REFERENCE_cn.md), section 14).

---

## Documentation Index

| Document | Content |
|----------|---------|
| [README_cn.md](README_cn.md) | Chinese version of this README |
| [tutorials/README_cn.md](tutorials/README_cn.md) | Two Jupyter tutorials |
| [docs/MAKE_TEST_GEMM_cn.md](docs/MAKE_TEST_GEMM_cn.md) | `make test_gemm` step-by-step |
| [docs/MAKE_ONNX_RUN_cn.md](docs/MAKE_ONNX_RUN_cn.md) | `make run` and `examples/onnx/` ONNX pipeline |
| [docs/nn_compiler_cn.md](docs/nn_compiler_cn.md) | Stage-1 NN compiler (`vta_backend.py`) workflow |
| [docs/main_vta_compiler_cn.md](docs/main_vta_compiler_cn.md) | Stage-2 VTA compiler (`main_vta_compiler.py`) workflow |
| [docs/vta_config_cn.md](docs/vta_config_cn.md) | `vta_config.json` field reference |
| [tutorials/tutorial2_operations_definition.py](tutorials/tutorial2_operations_definition.py) | Tutorial 2 script + **simulator step-by-step commands** (header comments) |
| [src/simulators/README_cn.md](src/simulators/README_cn.md) | Simulators overview (Chinese) |
| [src/simulators/functional_simulator/README_cn.md](src/simulators/functional_simulator/README_cn.md) | FSIM build/run (Chinese) |
| [docs/fsim_nn与fsim_single_layer_cn.md](docs/fsim_nn与fsim_single_layer_cn.md) | `fsim_nn` vs `fsim_single_layer` (Chinese) |
| [src/simulators/functional_simulator/README.md](src/simulators/functional_simulator/README.md) | FSIM English (paired with `README_cn.md`) |
| [docs/VTA_ISA_REFERENCE_cn.md](docs/VTA_ISA_REFERENCE_cn.md) | ISA and pseudocode reference |

---

## FAQ

**Q: README mentions `./vta_simulator` but it is not found?**  
A: Executables were renamed to `build/fsim_single_layer` (single layer) or `build/fsim_nn` (full network). Run `make build/fsim_single_layer` then `./build/fsim_single_layer`.

**Q: What is the difference between `make execute` and running the binary directly?**  
A: `make execute` links `build/fsim_single_layer` if needed, then runs it — equivalent to `make build/fsim_single_layer` followed by `./build/fsim_single_layer` from `functional_simulator/`.

**Q: I only ran `python3 tutorial2_operations_definition.py` and simulation fails with missing files?**  
A: Expected. Follow the [FSIM single-layer example](#fsim-single-layer-example-with-tutorial-instructions) above or section 6 of `tutorials/tutorial2_operations_definition.py`, or use TSIM `ComputeApp_lenet5_layer1`.

**Q: Does cycle-accurate simulation require JSON?**  
A: `ComputeApp_*` tests use binaries under `examples_compute/`; JSON is used by the separate `simulator/` test flow.
