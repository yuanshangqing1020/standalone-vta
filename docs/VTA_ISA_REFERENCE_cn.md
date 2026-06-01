# VTA 指令集参考手册（Standalone-VTA）

> **版本说明：** 本文档基于本仓库 `standalone-vta` 源码整理，与 Apache TVM-VTA 同源，但在编译器、仿真器与示例上有所扩展。  
> **主要参考源文件：**
> - `src/compiler/vta_compiler/operations_definition/structures.py`
> - `src/compiler/vta_compiler/operations_definition/README.md`
> - `src/simulators/functional_simulator/config/hw_spec.h`
> - `src/simulators/functional_simulator/config/hw_spec_const.h`
> - `config/vta_config.json`
> - `src/simulators/cycle_accurate_simulator/src/main/scala/core/ISA.scala`

---

## 目录

1. [架构概览](#1-架构概览)
2. [指令与微操作的两层结构](#2-指令与微操作的两层结构)
3. [顶层宏指令 OPCODE](#3-顶层宏指令-opcode)
4. [LOAD / STORE 指令](#4-load--store-指令)
5. [GEMM 指令](#5-gemm-指令)
6. [ALU 指令](#6-alu-指令)
7. [FINISH 指令](#7-finish-指令)
8. [微操作 UOP（32-bit）](#8-微操作-uop32-bit)
9. [片上存储与 BUFFER_ID](#9-片上存储与-buffer_id)
10. [依赖同步（Dependency Flags）](#10-依赖同步dependency-flags)
11. [默认硬件参数](#11-默认硬件参数)
12. [GEMM 计算语义](#12-gemm-计算语义)
13. [典型指令序列](#13-典型指令序列)
14. [指令编码与调试](#14-指令编码与调试)
15. [与编译器/教程的对应关系](#15-与编译器教程的对应关系)

---

## 1. 架构概览

VTA（Versatile Tensor Accelerator）是一个可编程张量加速器，通过 **128 位 CISC 宏指令** 驱动 **Load → Compute → Store** 流水线。

### 1.1 四大硬件模块

| 模块 | 职责 |
|------|------|
| **Fetch** | 从 DRAM 读取指令流，按 OPCODE 解码并分发到各模块的命令队列 |
| **Load** | 将输入（INP）、权重（WGT）等 2D 张量从 DRAM 搬运到片上 SRAM |
| **Compute** | 执行 GEMM（矩阵乘累加）与 ALU（激活/池化等逐元素运算）；同时负责加载 UOP 与 bias |
| **Store** | 将计算结果从片上 OUT/ACC buffer 写回 DRAM |

### 1.2 默认流水线

```
Load (INP/WGT/UOP/ACC) → Compute (GEMM / ALU) → Store (OUT)
```

可根据需要插入多条 Compute 指令，例如：

```
Load → GEMM → ALU(ReLU) → Store
Load → GEMM → ALU(ADD) → ALU(SHR) → Store   // 平均池化
```

### 1.3 CPU 如何启动 VTA

Fetch 模块通过三个 MMIO 寄存器由 CPU 编程（见 `operations_definition/README.md`）：

| 寄存器 | 方向 | 含义 |
|--------|------|------|
| `control` | 读/写 | 启动 Fetch；读取完成状态 |
| `insn_count` | 写 | 待执行指令条数 |
| `insns` | 写 | 指令流在 DRAM 中的起始物理地址 |

执行步骤：

1. CPU 在 DRAM 中准备物理连续指令缓冲区；
2. 写入 `insns`（起始地址）和 `insn_count`（长度）；
3. 置位 `control` 启动 VTA；
4. Fetch 经 DMA 读取指令并分发。

---

## 2. 指令与微操作的两层结构

VTA 采用 **宏指令 + 微操作（UOP）** 的两层设计：

| 层级 | 位宽 | 作用 |
|------|:----:|------|
| **Instruction（宏指令）** | 128 bit | 指定操作类型（LOAD/GEMM/ALU/…）、UOP 范围、循环次数、索引步进、DRAM/SRAM 地址等 |
| **UOP（微操作）** | 32 bit | 指定一次运算所访问的具体 buffer 索引（ACC/INP/WGT 或 DST/SRC） |

关系如下：

- 宏指令给出 UOP 索引区间 `[uop_bgn, uop_end)`（左闭右开）；
- 宏指令定义双层循环 `loop_out × loop_in`，以及各索引在每个循环上的步进因子；
- 每次内层循环迭代，依次执行区间内全部 UOP；
- **UOP 只描述数据访问模式，真正的运算类型由宏指令 OPCODE 决定（GEMM 或 ALU）。**

---

## 3. 顶层宏指令 OPCODE

所有宏指令共享 128 bit 格式，通过最低 3 bit 的 **OPCODE** 区分类型。

| OPCODE | 名称 | 二进制 | 十进制 | 功能 | 解码后结构体 |
|:------:|------|:------:|:------:|------|-------------|
| 0 | **LOAD** | 0b000 | 0 | 从 DRAM 加载到片上 SRAM | `VTAMemInsn` |
| 1 | **STORE** | 0b001 | 1 | 从片上 SRAM 写回 DRAM | `VTAMemInsn` |
| 2 | **GEMM** | 0b010 | 2 | 矩阵乘累加 | `VTAGemInsn` |
| 3 | **FINISH** | 0b011 | 3 | 终止指令流 | `VTAMemInsn`（仅 OPCODE 有效） |
| 4 | **ALU** | 0b100 | 4 | 张量 ALU 逐元素运算 | `VTAAluInsn` |

**Fetch 分发规则（摘要）：**

| 指令 | 目标模块 |
|------|----------|
| LOAD + INP/WGT | Load 模块 |
| LOAD + UOP/ACC | Compute 模块 |
| STORE | Store 模块 |
| GEMM / ALU | Compute 模块 |
| FINISH | Fetch（终止） |

**注意：** 指令流 **必须以 FINISH 结尾**。周期精确仿真器 JSON 测试中的 FINISH 十六进制示例为：

```
00000000000000000000000000000003
```

（小端序编码，最低字节 `0x03` 即 OPCODE=3。）

---

## 4. LOAD / STORE 指令

LOAD 与 STORE 共用 `VTAMemInsn` 结构，通过 OPCODE 区分方向。二者可在搬运 2D 张量的同时完成 **padding**（在 LOAD 路径上填零）。

### 4.1 位域定义（128 bit，小端）

#### 低 64 bit（字段 0）

| 字段 | 位宽 | 说明 |
|------|:----:|------|
| `opcode` | 3 | 0=LOAD，1=STORE |
| `pop_prev_dep` | 1 | 依赖队列：弹出前一阶段 token |
| `pop_next_dep` | 1 | 依赖队列：弹出后一阶段 token |
| `push_prev_dep` | 1 | 依赖队列：推送 token 到前一阶段 |
| `push_next_dep` | 1 | 依赖队列：推送 token 到后一阶段 |
| `buffer_id` | 3 | 目标/源 SRAM 类型（见 §9） |
| `sram_base` | 16 | 片上 SRAM 起始地址（以 buffer 元素为单位） |
| `dram_base` | 32 | DRAM 起始地址 |
| `unused` | 6 | 保留 |

#### 高 64 bit（字段 1）

| 字段 | 位宽 | 说明 |
|------|:----:|------|
| `y_size` | 16 | 从 DRAM 复制的行数 |
| `x_size` | 16 | 每行复制的列数（以 memory element 计） |
| `x_stride` | 16 | DRAM 行间步长（以 memory element 计） |
| `y_pad_top` | 4 | 顶部 padding 行数 |
| `y_pad_bottom` | 4 | 底部 padding 行数 |
| `x_pad_left` | 4 | 左侧 padding 列数 |
| `x_pad_right` | 4 | 右侧 padding 列数 |

> Python 结构体字段名（`structures.py`）与 C 头文件（`hw_spec.h`）对应关系：  
> `y_pad_top/bottom` ↔ `y_pad_0/y_pad_1`，`x_pad_left/right` ↔ `x_pad_0/x_pad_1`。

### 4.2 LOAD 伪代码

```python
for i in range(y_size + y_pad_top + y_pad_bottom):
    for j in range(x_size + x_pad_left + x_pad_right):
        sram_loc = sram_base + i * (x_size + x_pad_left + x_pad_right) + j
        dram_loc = dram_base + (i - y_pad_top) * x_stride + (j - x_pad_left)
        if (i < y_pad_top or i >= y_size + y_pad_top or
            j < x_pad_left or j >= x_size + x_pad_left):
            sram[sram_loc] = 0          # padding 区域填零
        else:
            sram[sram_loc] = dram[dram_loc]
```

STORE 为反向搬运，通常不再添加 padding。

### 4.3 模拟器别名（Chisel ISA.scala）

| 别名 | 含义 |
|------|------|
| `LUOP` | LOAD → UOP cache |
| `LWGT` | LOAD → WGT buffer |
| `LINP` | LOAD → INP buffer |
| `LACC` | LOAD → ACC buffer（bias/累加器初值） |
| `SOUT` | STORE ← OUT buffer |

---

## 5. GEMM 指令

GEMM 是 VTA 的核心计算指令：对 **INP buffer** 与 **WGT buffer** 中的张量块做乘累加，结果写入 **ACC 寄存器文件**。

### 5.1 位域定义（128 bit）

#### 低 64 bit（与 ALU 共用前缀）

| 字段 | 位宽 | 说明 |
|------|:----:|------|
| `opcode` | 3 | 固定为 2 |
| `pop_prev_dep` | 1 | 从 Load 阶段 pop token |
| `pop_next_dep` | 1 | 从 Store 阶段 pop token |
| `push_prev_dep` | 1 | 向 Load 阶段 push token |
| `push_next_dep` | 1 | 向 Store 阶段 push token |
| `reset` | 1 | **1 = 清空 ACC 寄存器**（GeMM 开始前常用） |
| `uop_bgn` | 13 | UOP 起始索引（含） |
| `uop_end` | 14 | UOP 结束索引（**不含**） |
| `loop_out` | 14 | 外层循环次数 |
| `loop_in` | 14 | 内层循环次数 |
| `unused` | 1 | 保留 |

#### 高 64 bit（GEMM 专用）

| 字段 | 位宽 | 说明 |
|------|:----:|------|
| `dst_factor_out` | 11 | 外层循环 ACC 索引步进（X1） |
| `dst_factor_in` | 11 | 内层循环 ACC 索引步进（X0） |
| `src_factor_out` | 11 | 外层循环 INP 索引步进（Y1） |
| `src_factor_in` | 11 | 内层循环 INP 索引步进（Y0） |
| `wgt_factor_out` | 10 | 外层循环 WGT 索引步进（Z1） |
| `wgt_factor_in` | 10 | 内层循环 WGT 索引步进（Z0） |

> C 头文件命名：`dst_*` ↔ ACC，`src_*` ↔ INP，`wgt_*` ↔ WGT。

### 5.2 执行伪代码

```python
for i0 in range(loop_out):
    for i1 in range(loop_in):
        for uop_idx in range(uop_bgn, uop_end):
            acc_idx, inp_idx, wgt_idx = decode_gemm_uop(uop_buffer[uop_idx])
            acc_idx += i1 * dst_factor_in + i0 * dst_factor_out
            inp_idx += i1 * src_factor_in + i0 * src_factor_out
            wgt_idx += i1 * wgt_factor_in + i0 * wgt_factor_out
            acc_mem[acc_idx] += dot(inp_mem[inp_idx], wgt_mem[wgt_idx])
```

其中 `dot(·,·)` 为 **16 元素 int8 向量点乘，累加到 int32 ACC**（见 §12）。

### 5.3 reset 位

- `reset=1`：在执行本条 GEMM 所描述的 UOP 循环之前，清空 Compute 内部 ACC 状态；
- LeNet-5 等示例中，第一条 GEMM 指令通常 `reset=1`，后续 GEMM 指令 `reset=0` 以继续累加。

---

## 6. ALU 指令

ALU 在 **ACC 寄存器文件** 上执行逐元素（张量级）运算，用于 ReLU、加法、移位（除法）、乘法等。

### 6.1 位域定义（128 bit）

低 64 bit 与 GEMM 相同（含 `uop_bgn/end`、`loop_out/in`），但 ALU 的 `reset` 位一般置 0。

高 64 bit（ALU 专用）：

| 字段 | 位宽 | 说明 |
|------|:----:|------|
| `dst_factor_out` | 11 | 外层循环目的 ACC 索引步进 |
| `dst_factor_in` | 11 | 内层循环目的 ACC 索引步进 |
| `src_factor_out` | 11 | 外层循环源 ACC 索引步进 |
| `src_factor_in` | 11 | 内层循环源 ACC 索引步进 |
| `alu_opcode` | 3 | ALU 运算类型（见下表） |
| `use_imm` | 1 | 0=张量-张量；1=张量-立即数 |
| `imm` | 16 | 立即数（有符号，允许负值） |

### 6.2 ALU 运算码

| alu_opcode | 名称 | 值 | use_imm=0（张量-张量） | use_imm=1（张量-标量） |
|:----------:|------|:--:|------------------------|------------------------|
| 0 | **MIN** | 0b000 | `R[dst] = min(R[dst], R[src])` | — |
| 1 | **MAX** | 0b001 | `R[dst] = max(R[dst], R[src])` | — |
| 2 | **ADD** | 0b010 | `R[dst] = R[dst] + R[src]` | `R[dst] = R[dst] + imm`（ADDI） |
| 3 | **SHR** | 0b011 | — | `R[dst] = R[dst] >> imm`（SHRI，算术右移） |
| 4 | **MUL** | 0b100 | `R[dst] = R[dst].lo * R[src].lo` | `R[dst] = R[dst].lo * imm`（MULI） |

**Chisel 模拟器别名：** `VMIN`、`VMAX`、`VADD`、`VSHX`（shift，含 SHR/SHL）。

### 6.3 SHL（左移）的实现方式

本仓库 Chisel 实现中，**左移 SHL 复用 SHR 编码**：当 `alu_opcode=3` 且 `imm` 的最高位为 1 时，解释为左移而非右移（见 `TensorAlu.scala`）。文档中 `structures.py` 注释写作 `4-MUL/SHL`，即 opcode 4 在特定配置下也可与移位相关，以仿真器行为为准。

### 6.4 执行伪代码

```python
for i0 in range(loop_out):
    for i1 in range(loop_in):
        for uop_idx in range(uop_bgn, uop_end):
            dst_idx, src_idx = decode_alu_uop(uop_buffer[uop_idx])
            dst_idx += i1 * dst_factor_in + i0 * dst_factor_out
            src_idx += i1 * src_factor_in + i0 * src_factor_out
            if use_imm:
                acc_mem[dst_idx] = ALU_OP(alu_opcode, acc_mem[dst_idx], imm)
            else:
                acc_mem[dst_idx] = ALU_OP(alu_opcode, acc_mem[dst_idx], acc_mem[src_idx])
```

### 6.5 常见用法

| 网络算子 | VTA 实现 |
|----------|----------|
| **ReLU** | `MAX(dst, 0)`：`alu_opcode=1`，`use_imm=1`，`imm=0` |
| **平均池化** | 多次 `ADD` 将邻域元素相加 + `SHRI` 除以窗口大小（整数除法，向下取整） |
| **偏置加法** | `ADD` 或先 `LACC` 加载 bias 再 GEMM 累加 |

---

## 7. FINISH 指令

| 属性 | 说明 |
|------|------|
| OPCODE | 3 |
| 结构体 | 使用 `VTAMemInsn` 解码路径，其余字段通常为 0 |
| 作用 | 通知 Fetch 模块指令流结束，VTA 执行完成 |
| 要求 | **每条指令序列的最后一条必须是 FINISH** |

---

## 8. 微操作 UOP（32-bit）

UOP 存储在 UOP cache 中，由 LOAD UOP 指令从 DRAM 加载。

### 8.1 结构（`VTAUop`，32 bit）

| 字段 | 位宽 | GEMM 语义 | ALU 语义 |
|------|:----:|-----------|----------|
| `dst_idx` | 11 | ACC 索引 | 目的 ACC 索引 |
| `src_idx` | 11 | INP 索引 | 源 ACC 索引 |
| `wgt_idx` | 10 | WGT 索引 | **未使用** |

> C 头文件 `VTAUop` 中 `src_idx` 位宽取 `max(LOG_ACC_BUFF_DEPTH, LOG_INP_BUFF_DEPTH)`，与 Python 固定 11+11+10 一致。

### 8.2 GEMM UOP 示例

一次 16×16 × 16×16 矩阵乘通常只需 1 条 UOP：

```
dst_idx=0, src_idx=0, wgt_idx=0
```

LeNet-5 conv1 需要多条 UOP 以覆盖多个 INP/WGT 块（见 `examples/lenet5/layer1.py`）。

### 8.3 ALU UOP 示例

ReLU 对 ACC[0] 做 MAX(·, 0)：

```
dst_idx=0, src_idx=0, wgt_idx=0   // wgt_idx 忽略
```

---

## 9. 片上存储与 BUFFER_ID

### 9.1 BUFFER_ID 编码（LOAD/STORE）

| buffer_id | 名称 | 值 | 用途 |
|:---------:|------|:--:|------|
| 0 | UOP cache | 0b000 | 存储 32-bit 微操作 |
| 1 | WGT buffer | 0b001 | 权重张量块 |
| 2 | INP buffer | 0b010 | 输入激活块 |
| 3 | ACC buffer | 0b011 | 累加器/偏置（int32） |
| 4 | OUT buffer | 0b100 | 输出缓冲（STORE 目标） |
| 5 | ACC 8-bit | 0b101 | 8-bit 累加器视图 |

### 9.2 各 Buffer 中「一个元素」的语义（周期精确仿真器）

以下为本仓库仿真器测试 JSON 中使用的 **实际数据布局**（与 `ComputeTest` README 一致）：

| Buffer | 一个索引单元的内容 | 典型位宽 |
|--------|-------------------|----------|
| INP | 16 个 int8（对应 16×16 块的一行） | 16 × 8 bit |
| WGT | 256 个 int8（完整 16×16 权重块） | 16×16 × 8 bit |
| ACC | 16 个 int32 | 16 × 32 bit |
| OUT | 16 个 int16（STORE 输出） | 16 × 16 bit |
| UOP | 1 条 32-bit 微操作 | 32 bit |

**物理地址计算（DRAM）：**

```
物理地址（字节）= base_addr + 逻辑索引 × 该向量字节宽度
```

示例（ACC 向量 64 字节/条）：

```
phys = acc_baddr + 0x0001 × 64 = 0x0040
```

---

## 10. 依赖同步（Dependency Flags）

VTA 通过 Load / Compute / Store 三阶段之间的 **依赖队列（semaphore）** 实现流水线握手，避免 buffer 冲突。

每条指令含 4 个 1-bit 依赖标志：

| 字段 | 含义 |
|------|------|
| `pop_prev_dep` | 从「前一阶段→Compute」队列弹出 token |
| `pop_next_dep` | 从「后一阶段→Compute」队列弹出 token |
| `push_prev_dep` | 向「Compute→前一阶段」队列推送 token |
| `push_next_dep` | 向「Compute→后一阶段」队列推送 token |

编译器内部维护四类计数（见 `instructions_template.py`）：

| 信号 | 方向 |
|------|------|
| `LD->CMP` | Load → Compute |
| `CMP->LD` | Compute → Load |
| `ST->CMP` | Store → Compute |
| `CMP->ST` | Compute → Store |

**NOP 指令：** 当某阶段需要等待依赖时，可插入 `y_size=0, x_size=0` 的 LOAD/STORE 空操作，仅更新依赖 token 而不搬运数据（`nop_stage_instruction()`）。

**典型执行阶段划分（README General remarks）：**

0. **Reset 阶段**：GEMM `reset=1` 清空 ACC；
1. **操作阶段**：LOAD / GEMM / ALU / STORE；
2. **收尾阶段**：排空依赖队列 + **FINISH**。

---

## 11. 默认硬件参数

参数定义于 [`config/vta_config.json`](config/vta_config.json)，编译时生成宏（如 `vta_config_def.txt`）。

### 11.1 当前仓库默认配置

| 配置项 | 值 | 推导结果 |
|--------|:--:|----------|
| `LOG_BLOCK` | 4 | **分块大小 = 2⁴ = 16**（16×16 GEMM 块） |
| `LOG_BATCH` | 0 | Batch = 1 |
| `LOG_INP_WIDTH` | 5 | 见 hw_spec 宏（仿真中 INP 仍按 int8 向量组织） |
| `LOG_WGT_WIDTH` | 5 | 同上 |
| `LOG_ACC_WIDTH` | 5 | ACC 元素 int32 |
| `LOG_UOP_BUFF_SIZE` | 15 | UOP cache 容量 = 2¹⁵ B |
| `LOG_INP_BUFF_SIZE` | 17 | INP buffer 容量 = 2¹⁷ B |
| `LOG_WGT_BUFF_SIZE` | 20 | WGT buffer 容量 = 2²⁰ B |
| `LOG_ACC_BUFF_SIZE` | 17 | ACC buffer 容量 = 2¹⁷ B |

**关键约束：** VTA GEMM core 每次处理 **BLOCK_IN × BLOCK_OUT = 16×16** 的 int8 矩阵块，累加到 int32 ACC。这也是 Tutorial 1 中将矩阵 padding 到 16 倍数的原因。

---

## 12. GEMM 计算语义

### 12.1 单次 GEMM 微操作

给定：

- `inp_mem[inp_idx]`：长度为 16 的 int8 向量（INP 块的一行）；
- `wgt_mem[wgt_idx]`：16×16 int8 权重块；
- `acc_mem[acc_idx]`：长度为 16 的 int32 累加向量；

则：

```
acc_mem[acc_idx][k] += Σ_{j=0}^{15} inp_mem[inp_idx][j] * wgt_mem[wgt_idx][j, k]
```

即：**一个 INP 行向量 × 一个 WGT 块 → 累加到 ACC 行向量**。完整 16×16 矩阵乘需要多个 INP 行 UOP 循环累加。

### 12.2 与 Im2Row 大矩阵的关系

大矩阵（如 LeNet-5 的 784×25）先在外部编译阶段分块为 16×16 子块，再通过 **一条或多条 GEMM 宏指令 + 多条 UOP + 嵌套循环** 覆盖全部块索引。这与 Tutorial 1 / `tutorial1_data_definition.py` 中的分块逻辑一致。

---

## 13. 典型指令序列

### 13.1 最小 16×16 矩阵乘

```
1. LOAD  UOP    (LUOP)
2. LOAD  INP    (LINP)
3. LOAD  WGT    (LWGT)
4. GEMM         (reset=1, uop 覆盖全部乘累加)
5. STORE OUT    (SOUT)
6. FINISH
```

### 13.2 16×16 GEMM + ReLU

```
...
4. GEMM         (reset=1)
5. LOAD  UOP    (ALU 用 UOP)
6. ALU  MAX     (use_imm=1, imm=0)
7. STORE OUT
8. FINISH
```

### 13.3 2×2 平均池化（示意）

对 4 元素向量做平均（整数除法）：

```
ALU ADD  (累加邻域元素，2 次)
ALU SHRI (imm=2，相当于 ÷4)
```

详见 `operations_definition/examples/insn_average_pooling.py`。

### 13.4 LeNet-5 第一层（conv1 + ReLU + AvgPool）

| 步骤 | 指令 | 说明 |
|:----:|------|------|
| 1 | LOAD UOP / INP / WGT | 加载微操作与数据 |
| 2 | GEMM (reset=1) | 784×25 × 25×6 分块乘，单条 GEMM + 多 UOP/循环 |
| 3 | ALU MAX(0) | ReLU |
| 4 | ALU ADD + SHR | 2×2 平均池化 |
| 5 | STORE OUT | 写回 |
| 6 | FINISH | 结束 |

参考：`operations_definition/examples/lenet5/layer1.py`、`insn_lenet5_conv1_relu_average_pooling.py`。

---

## 14. 指令编码与调试

### 14.1 字节序

- 指令为 **128 bit 小端（Little-Endian）** 结构体；
- Python 打印十六进制时按字节反转（`structures.py` 中 `hex_128bit`）。

### 14.2 解码工具

在 `operations_definition/` 目录下：

```bash
python structures.py
# 或于代码中调用：
from structures import decode_vta_insn, decode_uop
decode_vta_insn("00000000000000000000000000000003")  # FINISH
```

### 14.3 生成指令的 API

| 函数 | 文件 | 作用 |
|------|------|------|
| `load_store_instruction()` | `instructions_template.py` | 构造 LOAD/STORE |
| `gemm_instruction()` | 同上 | 构造 GEMM |
| `alu_instruction()` | 同上 | 构造 ALU |
| `nop_stage_instruction()` | 同上 | 依赖同步空操作 |

编译器自动生成见 `instructions_generator.py`、`step_instructions.py`。

---

## 15. 与编译器/教程的对应关系

| 概念 | 教程/脚本 | 指令层 |
|------|-----------|--------|
| Im2Row 矩阵 A、B | `tutorials/tutorial1_data_definition.py` Tutorial 1 | 编译后变为 INP/WGT buffer 内容 |
| 16×16 分块 | `data_definition.py` | GEMM 每次处理的块 |
| 矩阵乘 | `A × B` | 一条或多条 **GEMM** 指令 |
| ReLU | `np.maximum(x, 0)` | **ALU MAX**，imm=0 |
| 平均池化 | 滑动窗口均值 | **ALU ADD** + **ALU SHRI** |
| 写回 DRAM | binary 输出 | **STORE OUT** |

---

## 附录 A：指令速查表

### 宏指令 OPCODE

| 值 | 指令 |
|:--:|------|
| 0 | LOAD |
| 1 | STORE |
| 2 | GEMM |
| 3 | FINISH |
| 4 | ALU |

### BUFFER_ID

| 值 | Buffer |
|:--:|--------|
| 0 | UOP |
| 1 | WGT |
| 2 | INP |
| 3 | ACC |
| 4 | OUT |
| 5 | ACC 8-bit |

### ALU OPCODE

| 值 | 运算 |
|:--:|------|
| 0 | MIN |
| 1 | MAX |
| 2 | ADD / ADDI |
| 3 | SHRI |
| 4 | MUL / MULI |

---

## 附录 B：相关文件索引

| 路径 | 内容 |
|------|------|
| `src/compiler/vta_compiler/operations_definition/structures.py` | 指令/UOP C 结构体 Python 绑定 |
| `src/compiler/vta_compiler/operations_definition/instructions_template.py` | 指令构造模板 |
| `src/compiler/vta_compiler/operations_definition/instructions_generator.py` | 指令序列生成 |
| `src/compiler/vta_compiler/operations_definition/examples/` | 手工指令示例 |
| `src/simulators/functional_simulator/config/hw_spec.h` | C 侧位域定义与伪代码 |
| `src/simulators/cycle_accurate_simulator/src/main/scala/core/ISA.scala` | Chisel 解码 BitPat |
| `config/vta_config.json` | 可配置硬件参数 |
| `tutorials/tutorial2_operations_definition*.ipynb` | 指令生成教程 |

---

*文档维护：若硬件参数或位域与源码不一致，以 `structures.py` 与 `hw_spec.h` 为准。*
