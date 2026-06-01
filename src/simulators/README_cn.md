# VTA 仿真器

本目录包含两套 **互补** 的 VTA 仿真器（不是可互相替代的重复实现）：

| 子目录 | 简称 | 语言 | 用途 |
|--------|------|------|------|
| [`functional_simulator/`](functional_simulator/) | **FSIM** | C++ | **功能正确性**：执行 `instructions.bin`，快速得到正确的张量结果 |
| [`cycle_accurate_simulator/`](cycle_accurate_simulator/) | **TSIM** | Chisel/Scala | **周期精确**：时序、流水线、细粒度硬件行为（较慢） |

英文说明：[`README.md`](README.md)

---

## FSIM 的两个入口（共享同一 C++ 核心）

功能仿真器提供 **两个可执行文件**（旧名 `vta_simulator` 已废弃）：

- **`build/fsim_single_layer`** — 逐层独立运行 VTA，每层自带 `input*.bin`
- **`build/fsim_nn`** — 整网前向，依赖 `dependency.csv` 做层间调度

二者均链接 `sim_driver` / `sim_tlpp` / `virtual_memory`。

| 文档 | 内容 |
|------|------|
| [`functional_simulator/README_cn.md`](functional_simulator/README_cn.md) | 本目录 FSIM 编译与运行 |
| [`docs/fsim_nn与fsim_single_layer_cn.md`](../../docs/fsim_nn与fsim_single_layer_cn.md) | 仿真器全景、`fsim_nn` vs `fsim_single_layer`、完整流程 |

---

## TSIM（周期精确仿真）

通过 sbt 测试运行，例如 `cli.ComputeApp_lenet5_layer1`。许多流程与 FSIM 共用 `compiler_output/` 下的单层二进制（如 `make test_gemm`）。`src/test/resources/` 下的 JSON 驱动测试是 TSIM 的另一条路径。

详见 [`cycle_accurate_simulator/README.md`](cycle_accurate_simulator/README.md)。

---

## 容易误判为「仿真器」的组件

| 组件 | 实际角色 |
|------|----------|
| **`reference_onnx.py` / `check_bin.py`** | ONNX 金标生成与 `final_output.bin` 数值比对 |
| **`src/compiler/` 下编译器** | 生成 `compiler_output/`，**不**执行 VTA 指令 |

---

## 快速选型

| 目标 | 使用 |
|------|------|
| 16×16 GEMM 环境冒烟 | FSIM `fsim_single_layer` + TSIM `ComputeApp`（`make test_gemm`） |
| ONNX 整网数值验证 | FSIM `fsim_nn` + `check`（`make run`） |
| 周期级硬件调试 | TSIM `ComputeApp_*` |

---

## 相关文档（仓库根目录）

| 文档 | 内容 |
|------|------|
| [`README_cn.md`](../../README_cn.md) | 仓库总览、FSIM/TSIM 命令与 FAQ |
| [`docs/MAKE_TEST_GEMM_cn.md`](../../docs/MAKE_TEST_GEMM_cn.md) | `test_gemm` 逐步流程 |
| [`docs/MAKE_ONNX_RUN_cn.md`](../../docs/MAKE_ONNX_RUN_cn.md) | `make run` ONNX 整网流程 |
