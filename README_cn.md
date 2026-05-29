# standalone-vta

一个经过维护、统一并扩展的通用张量加速器（VTA）生态系统。

## 概述

本仓库通过以下方式解决了原始 VTA 项目的局限性：

* **统一仿真：** 为功能仿真器（C++，FSIM）和周期精确仿真器（CHISEL，TSIM）提供一致的输入格式（原始二进制文件 + CSV 元数据）。
* **扩展的周期精确仿真：** 针对 Compute / Load / Store 等子模块提供多组测试用例。
* **独立编译器：** 开源、不依赖 TVM 的 VTA 编译器，从 VTA IR（JSON）或教程脚本生成二进制文件。

VTA 面向 CNN 中的矩阵乘等张量运算；本项目侧重可复现的编译—仿真—校验流程。

## 仓库结构

| 路径 | 说明 |
|------|------|
| `tutorials/` | 教程 Notebook 与脚本（`0.py` 数据定义、`1.py` 指令定义） |
| `src/compiler/` | VTA 编译器（`data_definition`、`operations_definition`、`nn_compiler` 等） |
| `src/simulators/functional_simulator/` | C++ **功能仿真**（可执行文件 `build/fsim_single_layer`、`build/fsim_nn`） |
| `src/simulators/cycle_accurate_simulator/` | CHISEL **周期精确仿真**（sbt 测试，如 `ComputeApp_lenet5_layer1`） |
| `examples/` | 端到端示例（`make test_gemm`、`make compile_and_run` 等） |
| `compiler_output/` | 编译器/教程写出的二进制与 CSV（仿真器默认读取） |
| `config/vta_config.json` | VTA 硬件参数 |
| `VTA_ISA_REFERENCE_cn.md` | VTA 指令集与 ISA 中文参考 |
| `MAKE_TEST_GEMM_cn.md` | **`make test_gemm` 逐步流程**（产物、日志、FSIM/TSIM） |
| `environment_setup/standalone-vta.yml` | Conda 环境定义 |

## 环境依赖

### 1. 基础（教程与功能仿真）

* **g++**（支持 C++17）、**make**、**python3**
* **Conda 环境：**

```bash
conda env create -f environment_setup/standalone-vta.yml
conda activate standalone-vta
```

### 2. 周期精确仿真（TSIM，可选）

* **JDK 17**
* **sbt**（Scala Build Tool）
* **Verilator >= 5.x**（CHISEL 生成 Verilog 后仿真）

安装示例见下文「TSIM 环境」；国内使用 sbt 建议配置 Maven 镜像（原 README 中阿里云配置仍适用）。

---

## 快速开始

### A. 官方示例（推荐首次验证环境）

在仓库根目录 `standalone-vta/` 下：

```bash
conda activate standalone-vta
cd examples
make help          # 查看所有目标
make test_gemm     # 16×16 GEMM：VTA 编译 + FSIM + TSIM
```

成功后会出现 `compiler_output/`（二进制与 CSV）和 `log_output/`（日志）。逐步说明见 [`MAKE_TEST_GEMM_cn.md`](MAKE_TEST_GEMM_cn.md)。

### B. 教程脚本（LeNet-5 第一层）

```bash
cd tutorials
python3 0.py       # Tutorial 1：Im2Row、填充、分块（控制台验证，不写全量 bin）
python3 1.py       # Tutorial 2：UOP/insn 演示 + 写出 uop.bin、instructions.bin
```

`1.py` 文件头有**完整的模拟器执行步骤**（含 TSIM / FSIM 与 `compiler_output` 文件对照表）。

---

## 功能仿真器（FSIM）

路径：`src/simulators/functional_simulator/`

### 可执行文件（注意：不是 `vta_simulator`）

旧文档中的 `vta_simulator`、`functional_simulator` 已更名。当前 Makefile **实际生成**：

| 目标 | 可执行文件 | 用途 |
|------|------------|------|
| `make build/fsim_single_layer` | `build/fsim_single_layer` | **单层**：读 `compiler_output/`，按 `layers_name.csv` 执行一层 |
| `make build/fsim_nn` | `build/fsim_nn` | **整网**：多层串联与后处理 |
| `make execute` | （同上） | 若需则**先编译** `fsim_single_layer`，再**运行**该程序 |
| `make nn_execute` | （同上） | 编译并运行 `fsim_nn` |

> **说明：** `make all` 的目标名为 `build/functional_simulator`，但 Makefile 中**没有**该规则的链接配方，请勿依赖 `make all`。请使用上表中的显式目标。

### `make execute` 做了什么？

1. 检查并编译 `build/fsim_single_layer`（含 `vta_config.py`、仿真核心、`external_lib/tvm` 等）。
2. 在 `functional_simulator/` 目录下执行 `build/fsim_single_layer`。
3. 程序从 **`../../../compiler_output/`**（相对 `functional_simulator/`，即仓库根下的 `compiler_output/`）读取：
   * `layers_name.csv` — 层数与每层文件名后缀
   * `metadata{后缀}.csv` — 矩阵维度
   * `input{后缀}.bin`、`weight{后缀}.bin`、`accumulator{后缀}.bin`、`uop{后缀}.bin`、`instructions{后缀}.bin` 等
4. 调用 `VTADeviceRun()` 执行指令流，打印结果（默认 `doPrint = true`）。

**仅运行 `tutorials/1.py` 时**只会生成 `uop.bin` 与 `instructions.bin`，不足以直接跑 FSIM；还需数据类二进制与 CSV（见 `tutorials/1.py` 第六节，或从 `examples_compute/lenet5_layer1/` 同步）。

### FSIM 单层示例（配合教程指令）

在仓库根目录：

```bash
# 1) 生成指令（若尚未执行）
cd tutorials && python3 1.py && cd ..

# 2) 从 TSIM 测试资源复制数据 + 指令到 compiler_output
EX=src/simulators/cycle_accurate_simulator/src/test/resources/examples_compute/lenet5_layer1
cp "$EX"/input.bin "$EX"/weight.bin "$EX"/accumulator.bin "$EX"/out.bin \
   "$EX"/memory_addresses.csv compiler_output/
cp "$EX"/expected_out_sram.bin compiler_output/expected_out.bin
cp compiler_output/uop.bin compiler_output/instructions.bin   # 1.py 已生成则可省略

# 3) 单层 FSIM 所需的 CSV（空后缀 = 文件名不带 QLinearConv 等后缀）
printf 'nb_vta_ir,1,False\n0,,0\n' > compiler_output/layers_name.csv
printf 'Matrix (or Block Size),Nb rows,Nb columns,Is it square?\nBS,16,16,True\nA,784,32,True\nX,784,16,True\nY,0,0,True\nC,784,16,True\n' \
  > compiler_output/metadata.csv

# 4) 编译并运行
cd src/simulators/functional_simulator
make build/fsim_single_layer
./build/fsim_single_layer
# 或：make execute
```

---

## 周期精确仿真器（TSIM）

路径：`src/simulators/cycle_accurate_simulator/`

`ComputeTest.scala` 等测试直接使用 `src/test/resources/examples_compute/<用例>/` 下的二进制，**不需要** JSON 即可跑通（JSON 用于 `src/test/scala/simulator/` 下的其它用例）。

### LeNet-5 第一层（与 `1.py` / `insn_lenet5_layer1.py` 对齐）

```bash
# 仓库根目录
cd tutorials && python3 1.py && cd ..

cp compiler_output/uop.bin compiler_output/instructions.bin \
   src/simulators/cycle_accurate_simulator/src/test/resources/examples_compute/lenet5_layer1/

cd src/simulators/cycle_accurate_simulator
sbt "testOnly cli.ComputeApp_lenet5_layer1"
```

该资源目录已包含 `input.bin`、`weight.bin`、`accumulator.bin`、`memory_addresses.csv`、`expected_out_sram.bin` 等；通常只需替换你新生成的 `uop.bin` / `instructions.bin`。

### TSIM 环境（节选）

```bash
sudo apt update && sudo apt install -y openjdk-17-jdk
# sbt、Verilator 安装见本文件历史版本或官方文档；Verilator 需 >= 5.x
```

---

## `compiler_output/` 常见文件

| 文件 | 来源 | FSIM 单层 | TSIM `lenet5_layer1` |
|------|------|-----------|----------------------|
| `instructions.bin` | `operations_definition` / `1.py` | 需要 | 需要 |
| `uop.bin` | 同上 | 需要 | 需要 |
| `input.bin` / `weight.bin` | `data_definition` / 示例资源 | 需要 | 资源目录自带 |
| `accumulator.bin` | 同上 | 需要 | 资源目录自带 |
| `layers_name.csv` | VTA 编译器或手动 | 需要 | 不需要（测试内嵌路径） |
| `metadata*.csv` | 同上 | 需要 | 不需要 |
| `inputQLinearConv*.bin` 等 | `examples` 整网编译 | 多层 FSIM | — |

若目录中已有整网 ONNX 编译产物（`QLinearConv*.bin`），与 `tutorials/1.py` 无关；`1.py` 只会覆盖同目录下的 `uop.bin` 与 `instructions.bin`。

---

## 指令解析（调试）

不跑仿真、仅查看指令字段时：

```bash
cd src/compiler/vta_compiler/operations_definition
python3 structures.py
```

或在 Python 中：`from structures import decode_vta_insn, decode_uop`（见 `VTA_ISA_REFERENCE_cn.md` 第 14 节）。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [tutorials/README_cn.md](tutorials/README_cn.md) | 两个 Jupyter 教程说明 |
| [tutorials/1.py](tutorials/1.py) | Tutorial 2 脚本 + **模拟器逐步命令**（文件头注释） |
| [src/simulators/functional_simulator/README.md](src/simulators/functional_simulator/README.md) | FSIM 英文说明（部分命令已过时，以本文为准） |
| [VTA_ISA_REFERENCE_cn.md](VTA_ISA_REFERENCE_cn.md) | ISA 与伪代码 |

---

## 常见问题

**Q：README 里写的 `./vta_simulator` 找不到？**  
A：可执行文件已改名为 `build/fsim_single_layer`（单层）或 `build/fsim_nn`（整网）。请使用 `make build/fsim_single_layer` 后 `./build/fsim_single_layer`。

**Q：`make execute` 和直接运行可执行文件有什么区别？**  
A：`make execute` 会在需要时先完成链接，再运行 `build/fsim_single_layer`；效果等同于先 `make build/fsim_single_layer` 再 `./build/fsim_single_layer`（须在 `functional_simulator/` 目录下）。

**Q：只跑了 `python3 1.py`，仿真报错缺文件？**  
A：正常。请按上文「FSIM 单层示例」或 `tutorials/1.py` 第六节补齐 `input.bin` 等，或改用 TSIM 的 `ComputeApp_lenet5_layer1` 路径。

**Q：周期精确仿真是否必须用 JSON？**  
A：`ComputeApp_*` 类测试使用 `examples_compute/` 下的二进制；JSON 用于另一套 `simulator/` 测试流程。
