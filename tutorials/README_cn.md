# 教程

本目录提供两个 Jupyter Notebook 及配套 Python 脚本，旨在讲解 standalone-vta 的使用方法，并演示 `standalone-vta/src/compiler/vta_compiler` 中的数据生成与处理流程，以供 `standalone-vta/src/simulators` 使用。

**延伸阅读：** 卷积、GEMM、16×16 分块（tile）与 UOP 的完整原理说明见 [`../docs/gemm_tile_conv原理_cn.md`](../docs/gemm_tile_conv原理_cn.md)。

- [教程 1：数据定义（data_definition）](https://mybinder.org/v2/gh/onera/standalone-vta/main?urlpath=%2Fdoc%2Ftree%2Ftutorials%2Ftutorial1_data_definition.ipynb) — 脚本：[`tutorial1_data_definition.py`](tutorial1_data_definition.py)
- [教程 2：操作定义（operations_definition）](https://mybinder.org/v2/gh/onera/standalone-vta/main?urlpath=%2Fdoc%2Ftree%2Ftutorials%2Ftutorial2_operations_definition.ipynb) — 脚本：[`tutorial2_operations_definition.py`](tutorial2_operations_definition.py)

## 教程 1：使用 vta_compiler/data_definition

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/onera/standalone-vta/main?urlpath=%2Fdoc%2Ftree%2Ftutorials%2Ftutorial1_data_definition.ipynb)

本 Notebook 聚焦于 `data_definition` 文件夹，用于定义 VTA 所要处理的数据。

编译器的这一部分用于在 `standalone-vta/compiler_output` 目录中生成以下文件：
- 二进制文件：`input.bin`、`weight.bin`、`out_init.bin`、`expected_out.bin`、`expected_out_sram.bin`
- CSV 文件：`memory_addresses.csv`（包含各数据类型的基地址）

生成的二进制文件包含按照 VTA 要求编码后的数据。

运行等价脚本：`python3 tutorial1_data_definition.py`

## 教程 2：使用 vta_compiler/operations_definition

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/onera/standalone-vta/main?urlpath=%2Fdoc%2Ftree%2Ftutorials%2Ftutorial2_operations_definition.ipynb)

本 Notebook 聚焦于 `operation_definition` 文件夹，用于定义 UOP 缓冲区以及 VTA 执行多种操作所需的指令。

编译器的这一部分用于在 `standalone-vta/compiler_output` 目录中生成以下二进制文件：`uop.bin`、`instructions.bin`。

生成的二进制文件包含按照 CHISEL 可计算格式编码后的数据。

运行等价脚本：`python3 tutorial2_operations_definition.py`（文件头注释含完整 FSIM/TSIM 步骤）
