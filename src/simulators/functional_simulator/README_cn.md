# 功能仿真器（C++ / FSIM）

本目录是 VTA 的 **C++ 功能仿真器（FSIM）**：在虚拟 DRAM 上对 `instructions.bin` / `uop.bin` 与数据执行指令，得到 **数值正确** 的结果（不做周期级时序建模）。

> **命名说明：** 旧文档中的可执行文件 `./vta_simulator` 已拆分为两个入口。完整关系（含 TSIM、废弃名称）见 [`docs/fsim_nn与fsim_single_layer_cn.md`](../../../docs/fsim_nn与fsim_single_layer_cn.md) 第 0 节「仿真器全景」。

英文说明：[`README.md`](README.md)

---

## 编译产物

| Makefile 目标 | 可执行文件 | 作用 |
|---------------|------------|------|
| `make build/fsim_single_layer` | `build/fsim_single_layer` | 单层 / 多层 **独立** VTA 算子（`layers_name.csv`） |
| `make build/fsim_nn` | `build/fsim_nn` | **整网** 前向（`dependency.csv`、层间链式、CPU 算子） |
| `make execute` | （运行 `fsim_single_layer`） | 按需编译后执行单层 FSIM |
| `make nn_execute` | （运行 `fsim_nn`） | 按需编译后执行整网 FSIM |

**请勿使用** `make all` → `build/functional_simulator`：目标名存在但 **没有有效的链接规则**。请用上表中的显式目标。

两个可执行文件共享的后端（链接进同一套 `.o`）：

- `sim_driver.cc` — `VTADeviceRun()` 等设备 API
- `sim_tlpp.cc` — Load/Store/计算相关功能模型
- `virtual_memory.cc` — 虚拟 DRAM
- `simulator_functions.cc` — 读 bin/csv、整网 reshape 等

---

## 环境依赖

- 支持 **C++17** 的 g++、`make`、**Python 3**（运行 `config/vta_config.py`）
- 仓库根目录下已有 **`compiler_output/`**（本目录相对路径：`../../../compiler_output/`）

---

## 编译

在本目录下：

```bash
make build/fsim_single_layer   # 单层入口
make build/fsim_nn             # 整网入口
```

在 `examples/` 下（推荐端到端流程）：

```bash
make fsim_compile_single_layer   # → build/fsim_single_layer
make fsim_compile                # → build/fsim_nn
```

---

## 运行

**当前工作目录** 必须为 `functional_simulator/`（或通过 Makefile 中 `cd $(FSIM_DIR)` 的目标间接运行）。

### 单层（例如 `make test_gemm` 之后）

```bash
./build/fsim_single_layer
# 或：make execute
```

从 `compiler_output/` 读取：`layers_name.csv`、`metadata*.csv`、`input*.bin`、`weight*.bin`、`accumulator*.bin`、`uop*.bin`、`instructions*.bin` 等。默认将结果 **打印到终端**。

### 整网（例如 `make run` 之后）

```bash
./build/fsim_nn
# 或：make nn_execute
```

此外还需要 `dependency.csv`、`input_nn.bin`；并写出 **`final_output.bin`**。

---

## 典型输入文件（`compiler_output/`）

由 VTA 编译器（`main_vta_compiler.py`）和/或 NN 编译器（`vta_backend.py`）生成：

| 文件 | 单层 FSIM | 整网 FSIM |
|------|-----------|-----------|
| `instructions{后缀}.bin` | 是 | 是 |
| `uop{后缀}.bin` | 是 | 是 |
| `input{后缀}.bin` | 是（每层从磁盘读） | 预加载一般不读，运行时链式填充 |
| `weight{后缀}.bin` 等 | 是 | 是 |
| `layers_name.csv` | 是 | 是 |
| `dependency.csv` | 否 | 是 |
| `input_nn.bin` | 否 | 是 |
| `final_output.bin` | 否（不写） | 写出 |
| `expected_out.bin` | 可选，手工对比 | 可选 |

后缀示例：空（`input.bin`）或 `QLinearConv1`（`inputQLinearConv1.bin`）。

---

## 与 `vta_simulator` 的关系

| 旧称 | 现称 |
|------|------|
| `./vta_simulator`（单层场景） | `./build/fsim_single_layer` |
| 整网功能仿真入口（概念上） | `./build/fsim_nn` |
| 目录名 `functional_simulator` | 整个 FSIM **工程目录**，不是第三个 exe |

硬件指令语义由 **`sim_driver`** 实现；`fsim_single_layer` 与 `fsim_nn` 只是两种不同的 **main 编排**。

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [`docs/fsim_nn与fsim_single_layer_cn.md`](../../../docs/fsim_nn与fsim_single_layer_cn.md) | 仿真器全景、两入口对比、流程图与 FAQ |
| [`../README_cn.md`](../README_cn.md) | FSIM 与 TSIM 总览 |
| [`README_cn.md`](../../../README_cn.md) | 仓库级 FSIM/TSIM 命令与 FAQ |
| [`docs/MAKE_TEST_GEMM_cn.md`](../../../docs/MAKE_TEST_GEMM_cn.md) | `test_gemm` + 单层 FSIM |
| [`docs/MAKE_ONNX_RUN_cn.md`](../../../docs/MAKE_ONNX_RUN_cn.md) | `make run` + 整网 FSIM |
| [`docs/dependency_csv详解.md`](../../../docs/dependency_csv详解.md) | `dependency.csv` 字段（`fsim_nn` 读取） |
