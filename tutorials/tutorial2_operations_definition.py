"""
Tutorial 2 示例脚本：LeNet-5 第一层卷积的 VTA 指令与运算定义（operations_definition）

本文件由 tutorial2_operations_definition.ipynb 中的 Python 代码单元抽取合并而成。
对应教程目标：在 Tutorial 1（data_definition）已得到 Im2Row 分块矩阵 A、B 的前提下，
为 VTA 模拟器生成并理解「微操作 UOP + 主机指令 insn」如何驱动 GEMM、ReLU、平均池化与存取。

================================================================================
一、与 Tutorial 1 的关系
================================================================================
Tutorial 1（0.py / tutorial1_data_definition）解决的是：
  卷积 → Im2Row 矩阵 A(784×25)、B(25×6) → 填充 → 16×16 分块。

Tutorial 2 在此基础上解决的是：
  如何用 VTA 指令描述「在 SRAM 上对这些分块数据执行何种运算」。
  本脚本使用与 LeNet-5 C1 相同的矩阵尺寸（A: 784×25, B: 25×6），
  并演示 UOP/insn 缓冲区的填充方式，以及用 Python 伪代码模拟硬件执行语义。


================================================================================
二、VTA 两类缓冲区
================================================================================
(1) UOP 缓冲区（VTAUop，32 位）
    每条微操作指定三个索引：
      - dst_idx：累加器 ACC 中的写入位置
      - src_idx：输入 INP 或 ACC 中的读取位置
      - wgt_idx：权重 WGT 中的读取位置（仅 GEMM 使用）

(2) 指令缓冲区（128 位，分三种结构）
    - VTAMemInsn：LOAD / STORE / FINISH，在 DRAM 与 SRAM 间搬运数据
    - VTAGemInsn：分块矩阵乘（GEMM）
    - VTAAluInsn：ReLU(MAX)、ADD、SHR 等 ALU 运算

依赖标志（pop/push dep）用于协调 LOAD 与 COMPUTE 流水线，避免数据竞争。


================================================================================
三、LeNet-5 第一层在 VTA 上的运算流水线
================================================================================
数据规模（与 Tutorial 1 一致）：
  - 原始 A：784×25 → 填充后 784×32 → 98 个 16×16 INP 块
  - 原始 B：25×6   → 填充后 32×16  → 2 个 16×16 WGT 块
  - 输出 ACC：784×16（49 个输出块 × 每块 16 列有效通道）

执行顺序：

  阶段 0 — 配置
    初始化 uop_buffer、insn_buffer；生成分块后的 block_input_matrix、block_weight_matrix。

  阶段 1 — DRAM → SRAM（LOAD，指令 I0–I4）
    I0  加载 RESET 用 UOP
    I1  GEMM RESET：清空 ACC（49×16 块）
    I2  LOAD INP：1568 字（98×16）
    I3  LOAD WGT：2 个 WGT 块
    I4  LOAD 后续运算 UOP（GEMM×2 + ReLU + Pool×3）

  阶段 2 — GEMM（指令 I5，UOP 1–2）
    将每个 16×16 INP 块视为 16 个 (16×1) 行向量，与 WGT 块做矩阵乘并累加到 ACC。
    伪代码 insn_GEMM 按 loop_out=49、loop_in=16 及索引因子遍历所有块与行。

  阶段 3 — ReLU（指令 I6，UOP 3）
    对 ACC 每个元素做 max(0, x)，消除负值。

  阶段 4 — 平均池化 2×2（指令 I7–I9，UOP 4–6）
    两次 ADD：将相邻 2×2 区域求和（数据量减半再减半）
    一次 SHR(imm=2)：右移 2 位，等价除以 4，得到平均值
    目的：将 28×28 特征图在空间上缩小为 14×14。

  阶段 5 — SRAM → DRAM（STORE，指令 I10–I13）
    I10 STORE 输出到 DRAM
    I11–I12 流水线 NOP（依赖握手）
    I13 FINISH

  阶段 6 — 二进制导出（对应 notebook 最后一格 %run）
    运行 insn_lenet5_layer1.py，将 uop_buffer / insn_buffer 写入
    compiler_output/ 下的 uop.bin 与 instructions.bin，供功能/周期精确模拟器读取。


================================================================================
四、脚本内部结构（与 notebook 单元顺序一致）
================================================================================
  配置 → UOP0 → LOAD I0–I4 → 预览块 → GEMM → ReLU → ADD×2 → SHR → STORE → 生成 .bin


================================================================================
五、运行本脚本（生成 uop.bin / instructions.bin）
================================================================================
  cd standalone-vta/tutorials
  python3 1.py

产物目录：standalone-vta/compiler_output/
  - uop.bin           （约 28 字节 = 7 条 UOP，由 insn_lenet5_layer1.py 写出）
  - instructions.bin  （约 3376 字节，完整 LeNet-5 第一层指令流）

说明：本脚本前半段用 Python 伪代码演示 notebook 中的 UOP/insn 构造；末尾
run_binary_generation() 调用 insn_lenet5_layer1.py 写入上述两个文件（与
examples_compute/lenet5_layer1/ 中同名的参考二进制一致）。


================================================================================
六、用模拟器执行（端到端步骤）
================================================================================
重要：仅执行 1.py 只会得到「指令 + UOP」两个文件。模拟器还需要「数据」二进制：
input.bin、weight.bin、accumulator.bin、out.bin、expected_out*.bin、
memory_addresses.csv 等。下面给出与本教程（LeNet-5 第一层）最匹配的跑法。

--------------------------------------------------------------------------------
6.0 环境准备（首次使用前）
--------------------------------------------------------------------------------
  conda activate standalone-vta          # 见 environment_setup/standalone-vta.yml
  # 周期精确仿真 (TSIM) 还需：JDK 17、sbt、Verilator >= 5.x
  # 详见仓库根目录 README_cn.md「环境依赖」一节

--------------------------------------------------------------------------------
6.1 周期精确仿真 TSIM（推荐，与 insn_lenet5_layer1 对齐）
--------------------------------------------------------------------------------
测试用例：ComputeApp_lenet5_layer1
资源目录：src/simulators/cycle_accurate_simulator/src/test/resources/examples_compute/lenet5_layer1/

步骤（在仓库根目录 standalone-vta/ 下执行）：

  # ① 生成 / 更新指令（若已运行可跳过）
  cd tutorials && python3 1.py && cd ..

  # ② 将新生成的指令复制到 TSIM 测试资源目录
  cp compiler_output/uop.bin \
     compiler_output/instructions.bin \
     src/simulators/cycle_accurate_simulator/src/test/resources/examples_compute/lenet5_layer1/

  # ③ 编译并运行 CHISEL 周期精确仿真（首次较慢，需下载依赖）
  cd src/simulators/cycle_accurate_simulator
  sbt "testOnly cli.ComputeApp_lenet5_layer1"

说明：
  - lenet5_layer1/ 目录内已附带 input.bin、weight.bin、accumulator.bin、
    out.bin、expected_out_sram.bin、memory_addresses.csv，无需再由 1.py 生成。
  - 若只替换 uop.bin / instructions.bin，而保留原有数据文件，即可验证
    「同一组 INP/WGT 上，新指令序列能否跑通并比对 expected_out_sram.bin」。
  - 测试通过时 sbt 报告 Tests: succeeded。

解析指令（可选，人工查看字段）：
  cd src/compiler/vta_compiler/operations_definition
  python3 structures.py
  # 或在 Python 中：from structures import decode_vta_insn, decode_uop

--------------------------------------------------------------------------------
6.2 功能仿真 FSIM（C++，需完整 compiler_output 布局）
--------------------------------------------------------------------------------
功能仿真入口：src/simulators/functional_simulator/build/fsim_single_layer
读取目录：standalone-vta/compiler_output/（相对 build 目录 ../../../compiler_output）

与 1.py 直接配套的「单层」跑法：先把 examples_compute/lenet5_layer1/ 中
全部二进制与 CSV 同步到 compiler_output，并改名为 FSIM 期望的「带后缀」文件名。
（单层教程后缀为空字符串时，文件名为 input.bin、uop.bin 等。）

  # ① 生成指令
  cd tutorials && python3 1.py && cd ..

  # ② 同步测试数据 + 指令到 compiler_output（覆盖 uop / instructions）
  EX=src/simulators/cycle_accurate_simulator/src/test/resources/examples_compute/lenet5_layer1
  cp $EX/input.bin $EX/weight.bin $EX/accumulator.bin $EX/out.bin \
     $EX/memory_addresses.csv compiler_output/
  cp $EX/expected_out_sram.bin compiler_output/expected_out.bin
  cp compiler_output/uop.bin compiler_output/instructions.bin  # 已由 1.py 生成

  # ③ 写入 FSIM 所需的 layers_name.csv / metadata.csv（单层、空后缀）
  printf 'nb_vta_ir,1,False\n0,,0\n' > compiler_output/layers_name.csv
  printf 'Matrix (or Block Size),Nb rows,Nb columns,Is it square?\nBS,16,16,True\nA,784,32,True\nX,784,16,True\nY,0,0,True\nC,784,16,True\n' \
    > compiler_output/metadata.csv

  # ④ 编译并运行
  cd src/simulators/functional_simulator
  make build/fsim_single_layer
  ./build/fsim_single_layer
  # 或：make execute

更完整的 FSIM 流水线（ONNX → 多层）：在 examples/ 目录执行 make test_gemm 或
make compile_and_run，由 vta_compiler 自动生成带后缀的 inputQLinearConv1.bin 等；
与仅运行 tutorials/1.py 的路径不同。

--------------------------------------------------------------------------------
6.3 产物与模拟器输入对照表
--------------------------------------------------------------------------------
  文件                      | 1.py 是否生成 | TSIM (lenet5_layer1) | FSIM (单层)
  --------------------------|---------------|----------------------|-------------
  uop.bin                   | 是            | 需要                 | 需要
  instructions.bin          | 是            | 需要                 | 需要
  input.bin                 | 否            | 资源目录自带         | 需复制/生成
  weight.bin                | 否            | 资源目录自带         | 需复制/生成
  accumulator.bin           | 否            | 资源目录自带         | 需复制/生成
  memory_addresses.csv      | 否            | 资源目录自带         | 需复制/生成
  expected_out*.bin         | 否            | 用于比对             | 可选

--------------------------------------------------------------------------------
6.4 常见问题
--------------------------------------------------------------------------------
  Q: 只有 uop.bin / instructions.bin，直接跑 FSIM 报错？
  A: 正常。请先按 6.1（TSIM）或 6.2 补齐数据文件。

  Q: compiler_output 里还有很多 QLinearConv*.bin？
  A: 那是 examples/make nn_compiler 整网编译产物，与 tutorials/1.py 无关；
     1.py 只会覆盖同目录下的 uop.bin 与 instructions.bin。

  Q: 周期精确仿真 README 写需要 JSON？
  A: ComputeApp_lenet5_layer1 使用二进制 + CSV，可直接按 6.1 运行；
     JSON 路径用于 src/test/scala/simulator/ 下的其它测试。


================================================================================
七、依赖模块
================================================================================
  - src/compiler/vta_compiler/operations_definition/structures.py
  - src/compiler/vta_compiler/data_definition/data_definition.py
  - src/compiler/vta_compiler/operations_definition/examples/insn_lenet5_layer1.py
"""

import sys
from pathlib import Path

import numpy as np

TUTORIAL_DIR = Path(__file__).resolve().parent
sys.path.append(str(TUTORIAL_DIR / "../src/compiler/vta_compiler/operations_definition"))
sys.path.append(str(TUTORIAL_DIR / "../src/compiler/vta_compiler/data_definition"))

import structures as structures_insn_uop
from data_definition import matrix_creation as _matrix_creation_zeros
from data_definition import matrix_padding, matrix_splitting


def matrix_creation(n_row, n_col, isInitRandom=False, random_bound=4, dtype=np.int8):
    """兼容 notebook 中 matrix_generator.matrix_creation 的随机矩阵接口。"""
    if isInitRandom:
        return np.random.randint(-random_bound, random_bound, size=(n_row, n_col), dtype=dtype)
    return _matrix_creation_zeros(m_row=n_row, n_col=n_col, dtype=dtype)


# notebook 中的 MG / MS 别名
import sys as _sys
MG = _sys.modules[__name__]
MS = _sys.modules[__name__]


# ==============================================================================
# 配置（CONFIGURATION）
# ==============================================================================
"""CONFIGURATION"""

# PACKAGE IMPORT
# --------------

# Parent folder

# UOP DEFINITION
# --------------
# Define empty UOP buffer
uop_buffer = []

# INSTRUCTION DEFINITION
# ----------------------
# Define empty instruction buffer
insn_buffer = []

# INPUT DATA
# --------------
block_size = 16

# A matrix size
A_row = 784
A_col = 25

block_input_matrix, _, _ = MS.matrix_splitting(MG.matrix_padding(MG.matrix_creation(n_row=A_row, n_col=A_col, isInitRandom=True, random_bound=4)), block_size, isWeight=False, isSquare=True)

# B matrix size
B_row = A_col # Required by matrix multiplication
B_col = 6

block_weight_matrix, _, _ = MS.matrix_splitting(MG.matrix_padding(MG.matrix_creation(n_row=B_row, n_col=B_col, isInitRandom=True, random_bound=4)), block_size, isWeight=True, isSquare=True)

# ==============================================================================
# UOP 0：RESET
# ==============================================================================
if (len(uop_buffer) < 1):
    uop_buffer.append(structures_insn_uop.VTAUop( # UOP 0 - reset
        dst_idx=0, 
        src_idx=0,
        wgt_idx=0
    ))

# ==============================================================================
# LOAD 指令 I0–I4（DRAM → SRAM）
# ==============================================================================
if (len(insn_buffer) < 1):
    
# Loading the RESET UOP

    insn_buffer.append(structures_insn_uop.VTAMemInsn( # I0: LOAD UOP
        opcode=0, # 0-LOAD, 1-STORE, 3-FINISH
        # DEP FLAG
        pop_prev_dep=0,
        pop_next_dep=0,
        push_prev_dep=0,
        push_next_dep=0,
        # Memory interaction
        buffer_id=0, # 0-UOP, 1-WGT, 2-INP, 3-ACC, 4-OUT, 5-ACC8bit
        sram_base=0x0000,
        dram_base=0x00001000,
        unused=0, # UNUSED
        # Operation over the data
        y_size=1,
        x_size=1,
        x_stride=1,
        y_pad_top=0,
        y_pad_bottom=0,
        x_pad_left=0,
        x_pad_right=0
    ))

# The ACC matrix is wiped, in case of RESET

    insn_buffer.append(structures_insn_uop.VTAGemInsn( # I1: GEMM RESET
        opcode=2, # 2-GEMM
        # DEP FLAG
        pop_prev_dep=0,
        pop_next_dep=0,
        push_prev_dep=1, # Ready signal to LOAD
        push_next_dep=0,
        # Operations
        reset=1, # 0-no, 1-reset
        uop_bgn=0, # UOP 0
        uop_end=1,
        loop_out=49, # Number of (16 x 16) blocks in ACC
        loop_in=16,  # Block size
        # UNUSED
        unused=0, # UNUSED
        # Index factors
        dst_factor_out=16, # Block size
        dst_factor_in=1,
        src_factor_out=0,
        src_factor_in=0,
        wgt_factor_out=0,
        wgt_factor_in=0
    ))
    
# Loading INP

    insn_buffer.append(structures_insn_uop.VTAMemInsn( # I2: LOAD INP
        opcode=0, # 0-LOAD, 1-STORE, 3-FINISH
        # DEP FLAG
        pop_prev_dep=0,
        pop_next_dep=1, # Acknowledge COMPUTE ready signal
        push_prev_dep=0,
        push_next_dep=0,
        # Memory interaction
        buffer_id=2, # 0-UOP, 1-WGT, 2-INP, 3-ACC, 4-OUT, 5-ACC8bit
        sram_base=0x0000,
        dram_base=0x00000100,
        unused=0, # UNUSED
        # Operation over the data
        y_size=1,
        x_size=1568, # Load 98*16 INP
        x_stride=1568,
        y_pad_top=0,
        y_pad_bottom=0,
        x_pad_left=0,
        x_pad_right=0
    ))
    
# Loading WGT

    insn_buffer.append(structures_insn_uop.VTAMemInsn( # I3: LOAD WGT
        opcode=0, # 0-LOAD, 1-STORE, 3-FINISH
        # DEP FLAG
        pop_prev_dep=0,
        pop_next_dep=0,
        push_prev_dep=0,
        push_next_dep=1, # Ready signal to COMPUTE
        # Memory interaction
        buffer_id=1, # 0-UOP, 1-WGT, 2-INP, 3-ACC, 4-OUT, 5-ACC8bit
        sram_base=0x0000,
        dram_base=0x00000020,
        unused=0, # UNUSED
        # Operation over the data
        y_size=1,
        x_size=2, # Load 2 WGT
        x_stride=2,
        y_pad_top=0,
        y_pad_bottom=0,
        x_pad_left=0,
        x_pad_right=0
    ))
    
# Loading UOPs for GEMM & Average Pooling operations

    insn_buffer.append(structures_insn_uop.VTAMemInsn( # I4: LOAD UOP
        opcode=0, # 0-LOAD, 1-STORE, 3-FINISH
        # DEP FLAG
        pop_prev_dep=1, # Acknowledge LOAD ready signal
        pop_next_dep=0, 
        push_prev_dep=0,
        push_next_dep=0,
        # Memory interaction
        buffer_id=0, # 0-UOP, 1-WGT, 2-INP, 3-ACC, 4-OUT, 5-ACC8bit
        sram_base=0x0001,
        dram_base=0x00001001,
        unused=0, # UNUSED
        # Operation over the data
        y_size=1,
        x_size=6, # Load 6 UOP (2 GeMM + 1 ReLU + 3 Pool)
        x_stride=6,
        y_pad_top=0,
        y_pad_bottom=0,
        x_pad_left=0,
        x_pad_right=0
    ))

# ==============================================================================
# 预览分块数据 A@0、B@0、C@0
# ==============================================================================
# ----------------------
# Splitting (16 x 16) block INP matrices into (16 x 1) vectors

# Input Matrix INP
# 首个 INP 块的第一行（16×1 行向量；notebook 原 API 为 block[0][0]）
_inp_row0 = block_input_matrix[0][0:1, :].T
print("First vector of first block of INP matrix (", _inp_row0.shape[0], " x ", 1, ")")
print(_inp_row0, "A@0")

# Weight Matrix WGT
print("x \nFirst block of WGT matrix (", block_weight_matrix[0].shape[0], " x ", block_weight_matrix[0].shape[1], ")")
print(block_weight_matrix[0], "B@0")

# Output Matrix ACC
C_0 = np.zeros((1, block_size))
print("= \nFirst vector of first block of ACC [to be filled] (", block_size, " x ", 1, ")")
print(C_0, "C@0")

# ==============================================================================
# GEMM（指令 I5，UOP 1–2）
# ==============================================================================
"""GEMM"""

# Generating the instructions for the GeMM, using A vectorized and B.

# ----------------------
# Defining the GEMM UOP buffer

if (len(uop_buffer) < 1 + 1):
    uop_buffer.append(structures_insn_uop.VTAUop( # UOP 1 - GEMM 0
        dst_idx=0, 
        src_idx=0,
        wgt_idx=0
    ))

if (len(uop_buffer) < 2 + 1):
    uop_buffer.append(structures_insn_uop.VTAUop( # UOP 2 - GEMM 1
        dst_idx=0, 
        src_idx=16,
        wgt_idx=1
    ))

# ----------------------
# Defining the GEMM Instruction buffer

index_insn = 5 # Instruction index

if (len(insn_buffer) < index_insn + 1):
    insn_buffer.append(structures_insn_uop.VTAGemInsn( # I5: GEMM
        opcode=2, # 2-GEMM
        # DEP FLAG
        pop_prev_dep=0,
        pop_next_dep=0,
        push_prev_dep=0,
        push_next_dep=0, 
        # Operations
        reset=0, # 0-no, 1-reset
        uop_bgn=1, # UOP 1 + UOP 2
        uop_end=3,
        loop_out=49,
        loop_in=16,
        # UNUSED
        unused=0, # UNUSED
        # Index factors
        dst_factor_out=16,
        dst_factor_in=1,
        src_factor_out=32,
        src_factor_in=1,
        wgt_factor_out=0,
        wgt_factor_in=0
    ))

# ----------------------
# Print the buffers
 
# Printing UOP Buffer
def print_uop_buffer(OP, uop_bgn, uop_end) :
    print(OP, "UOP BUFFER\nACC  INP  WGT\n")
    for i in range(uop_bgn, uop_end):
        print(uop_buffer[i].dst_idx, "  ", uop_buffer[i].src_idx, "  ", uop_buffer[i].wgt_idx, "\n")

# Printing ALU Instruction Buffer      
def print_insn_buffer_ALU(n_insn, OP):
    print(OP, "INSTRUCTIONS\nLP_OUT  LP_IN  DST_OUT  DST_IN  SRC_OUT  SRC_IN  OPCODE  IMM\n")
    print(insn_buffer[n_insn].loop_out, "     ", insn_buffer[n_insn].loop_in, "     ", insn_buffer[n_insn].dst_factor_out, "     ", insn_buffer[n_insn].dst_factor_in, "     ", 
          insn_buffer[n_insn].src_factor_out, "     ", 
          insn_buffer[n_insn].src_factor_in, "     ", insn_buffer[n_insn].opcode, "    ", insn_buffer[n_insn].imm)
    
# ----------------------
# Defining GEMM operation

def GEMM(A, B):
#    assert(A.shape[1] == B.shape[0])
    A = np.array(A)
    B = np.array(B)
    return A @ B

# ----------------------
# Pseudo-code GEMM

def insn_GEMM(ACC, WGT, INP):
    for i0 in range(insn_buffer[index_insn].loop_in):
        for i1 in range(insn_buffer[index_insn].loop_out):
            for uop_index in range(insn_buffer[index_insn].uop_bgn, insn_buffer[index_insn].uop_end):
                X, Y, Z = uop_buffer[uop_index].dst_idx, uop_buffer[uop_index].src_idx, uop_buffer[uop_index].wgt_idx
                dst_idx = i0 * insn_buffer[index_insn].dst_factor_in + i1 * insn_buffer[index_insn].dst_factor_out + X # Index ACC
                inp_idx = i0 * insn_buffer[index_insn].src_factor_in + i1 * insn_buffer[index_insn].src_factor_out + Y # Index INP
                wgt_idx = i0 * insn_buffer[index_insn].wgt_factor_in + i1 * insn_buffer[index_insn].wgt_factor_out + Z # Index WGT
                ACC[dst_idx] += GEMM(INP[inp_idx], WGT[wgt_idx])                                                       # Storage of GEMM(A, B) in ACC
    return ACC

# ----------------------
# Printing the data
# ----------------------

# Printing GEMM UOP Buffer
print_uop_buffer("GEMM", insn_buffer[index_insn].uop_bgn, insn_buffer[index_insn].uop_end)

# Printing GEMM Instruction Buffer 
print("GEMM INSTRUCTIONS\nLP_OUT  LP_IN  DST_OUT  DST_IN  SRC_OUT  SRC_IN  WGT_OUT  WGT_IN\n")
print(insn_buffer[index_insn].loop_out, "     ", insn_buffer[index_insn].loop_in, "     ", insn_buffer[index_insn].dst_factor_out, "     ", insn_buffer[index_insn].dst_factor_in, "     ", 
        insn_buffer[index_insn].src_factor_out, "     ", 
        insn_buffer[index_insn].src_factor_in, "     ", insn_buffer[index_insn].wgt_factor_out, "     ", insn_buffer[index_insn].wgt_factor_in, "\n")

# Printing the Output Matrix
INP_stack = np.vstack(block_input_matrix)       # Stacking the 98 (16 x 16) blocks of A
ACC = np.zeros((A_row, block_size))             # Initializing the Output Matrix C (49 blocks of size (16 x 16) stacked) with zeros

ACC_GEMM = insn_GEMM(ACC, block_weight_matrix, INP_stack)
#assert(ACC_GEMM[0] == block_output_matrix[0][0])
print("ACC - Output matrix post-GEMM (", ACC_GEMM.shape[0], "x", ACC_GEMM.shape[1], ")")
print(ACC_GEMM)

# ==============================================================================
# ReLU（指令 I6，UOP 3）
# ==============================================================================
"""ReLU ACTIVATION"""

# In data_definitions/user_configuration.py, if `useReLU=True` :

# ----------------------
# Defining the ALU-RELU UOP buffer

if (len(uop_buffer) < 3 + 1):
    uop_buffer.append(structures_insn_uop.VTAUop( # UOP 3 - ALU (relu)
        dst_idx=0, 
        src_idx=0,
        wgt_idx=0
    ))

# ----------------------
# Defining the ALU-RELU Instruction buffer

index_insn = 6 # Instruction index

if (len(insn_buffer) < index_insn + 1):
    insn_buffer.append(structures_insn_uop.VTAAluInsn( # I6: ALU - MAX IMM 0 (relu)
        opcode=4, # 4-ALU
        # DEP FLAG
        pop_prev_dep=0,
        pop_next_dep=0,
        push_prev_dep=0,
        push_next_dep=0,
        # Operations
        reset=0, # 0-no, 1-reset
        uop_bgn=3, # UOP 3
        uop_end=4,
        loop_out=49,
        loop_in=16,
        # UNUSED
        unused=0, # UNUSED
        # Index factors
        dst_factor_out=16,
        dst_factor_in=1, # ACC incremented by 1
        src_factor_out=16,
        src_factor_in=1, # INP incremented by 1
        alu_opcode=1, # 0-MIN, 1-MAX, 2-ADD, 3-SHR, 4-MUL
        use_imm=1, # 0-no, 1-yes
        imm=0
    ))

# ----------------------
# Defining RELU operation
def RELU(A, useReLU):
    if (useReLU):
        A = np.maximum(A, 0)
    return A

# ----------------------
# Pseudo-code ALU RELU

def insn_RELU(ACC):
    for i0 in range(insn_buffer[index_insn].loop_in):
        for i1 in range(insn_buffer[index_insn].loop_out):
            for uop_index in range(insn_buffer[index_insn].uop_bgn, insn_buffer[index_insn].uop_end):
                X = uop_buffer[uop_index].dst_idx
                dst_idx = i0 * insn_buffer[index_insn].dst_factor_in + i1 * insn_buffer[index_insn].dst_factor_out + X # Index ACC
                ACC[dst_idx] = RELU(ACC[dst_idx], True) # For every row of ACC, we do max(0, value) for each value of the row
    return ACC

# ----------------------
# Printing the data
# ----------------------

# Printing ReLU UOP Buffer
print_uop_buffer("RELU", insn_buffer[index_insn].uop_bgn, insn_buffer[index_insn].uop_end)

# Printing ReLU Instruction Buffer 
print_insn_buffer_ALU(index_insn, "RELU")

# Printing the Output Matrix

ACC_ReLU = insn_RELU(ACC_GEMM)
print("\nACC - Output matrix post-ReLU (", ACC_ReLU.shape[0], "x", ACC_ReLU.shape[1], ")")
print(ACC_ReLU)

# ==============================================================================
# 平均池化 ADD #1（指令 I7，UOP 4）
# ==============================================================================
"""AVERAGE POOLING - First ADD"""

# After this step, the relevant data storage is divided by two.

# ----------------------
# Defining the ADD #1 UOP buffer

if (len(uop_buffer) < 4 + 1):
    uop_buffer.append(structures_insn_uop.VTAUop( # UOP 4 - ALU (first add)
        dst_idx=0, 
        src_idx=1,
        wgt_idx=0
    ))

# ----------------------
# Defining the ADD #1 Instruction buffer

index_insn = 7 # Instruction index

if (len(insn_buffer) < index_insn + 1):
    insn_buffer.append(structures_insn_uop.VTAAluInsn( # I7: ALU - ADD (Average Pooling 1/3)
        opcode=4, # 4-ALU
        # DEP FLAG
        pop_prev_dep=0,
        pop_next_dep=0,
        push_prev_dep=0,
        push_next_dep=0,
        # Operations
        reset=0, # 0-no, 1-reset
        uop_bgn=4, # UOP 4
        uop_end=5,
        loop_out=1,
        loop_in=392,
        # UNUSED
        unused=0, # UNUSED
        # Index factors
        dst_factor_out=0,
        dst_factor_in=2, 
        src_factor_out=0,
        src_factor_in=2, 
        alu_opcode=2, # 0-MIN, 1-MAX, 2-ADD, 3-SHR, 4-MUL
        use_imm=0, # 0-no, 1-yes
        imm=0
    ))

# ----------------------
# Define ADD operation

def ADD(A, B):
    A = np.array(A)
    B = np.array(B)
    return A + B
        
# ----------------------
# Pseudo-code ALU ADD

def insn_ADD(ACC):
    for i0 in range(insn_buffer[index_insn].loop_in):
        for i1 in range(insn_buffer[index_insn].loop_out):
            for uop_index in range(insn_buffer[index_insn].uop_bgn, insn_buffer[index_insn].uop_end):
                X, Y = uop_buffer[uop_index].dst_idx, uop_buffer[uop_index].src_idx
                dst_idx = i0 * insn_buffer[index_insn].dst_factor_in + i1 * insn_buffer[index_insn].dst_factor_out + X
                inp_idx = i0 * insn_buffer[index_insn].src_factor_in + i1 * insn_buffer[index_insn].src_factor_out + Y
                ACC[dst_idx] = ADD(ACC[dst_idx], ACC[inp_idx])
    return ACC

# ----------------------
# Printing the data
# ----------------------

# Printing ADD #1 UOP Buffer
print_uop_buffer("ADD #1", insn_buffer[index_insn].uop_bgn, insn_buffer[index_insn].uop_end)

# Printing ADD #1 Instruction Buffer 
print_insn_buffer_ALU(index_insn, "ADD #1")

# Printing the Output Matrix
ACC_ADD1 = insn_ADD(ACC_ReLU)
print("\nACC - Output matrix post-first ADD (", ACC_ADD1.shape[0], "x", ACC_ADD1.shape[1], ")")
print(ACC_ADD1)

# ==============================================================================
# 平均池化 ADD #2（指令 I8，UOP 5）
# ==============================================================================
"""AVERAGE POOLING - Second ADD"""

# After this step, the relevant data storage is divided by two. (4 total)

# ----------------------
# Defining the ADD #2 UOP buffer

if (len(uop_buffer) < 5 + 1):
    uop_buffer.append(structures_insn_uop.VTAUop( # UOP 5 - ALU (second add)
        dst_idx=0, 
        src_idx=28,
        wgt_idx=0
    ))

# ----------------------
# Defining the ADD #2 Instruction buffer

index_insn = 8 # Instruction index

if (len(insn_buffer) < index_insn + 1):
    insn_buffer.append(structures_insn_uop.VTAAluInsn( # I8: ALU - ADD (Average Pooling 2/3)
        opcode=4, # 4-ALU
        # DEP FLAG
        pop_prev_dep=0,
        pop_next_dep=0,
        push_prev_dep=0,
        push_next_dep=0,
        # Operations
        reset=0, # 0-no, 1-reset
        uop_bgn=5, # UOP 5
        uop_end=6,
        loop_out=14,
        loop_in=14,
        # UNUSED
        unused=0, # UNUSED
        # Index factors
        dst_factor_out=56,
        dst_factor_in=2, 
        src_factor_out=56,
        src_factor_in=2, 
        alu_opcode=2, # 0-MIN, 1-MAX, 2-ADD, 3-SHR, 4-MUL
        use_imm=0, # 0-no, 1-yes
        imm=0
    ))

# ----------------------
# Printing the data
# ----------------------

# Printing ADD #2 UOP Buffer
print_uop_buffer("ADD #2", insn_buffer[index_insn].uop_bgn, insn_buffer[index_insn].uop_end)

# Printing ADD #2 Instruction Buffer 
print_insn_buffer_ALU(index_insn, "ADD #2")

# Printing the Output Matrix
ACC_ADD2 = insn_ADD(ACC_ADD1)
print("\nACC - Output matrix post-second ADD (", ACC_ADD2.shape[0], "x", ACC_ADD2.shape[1], ")")
print(ACC_ADD2)

# ==============================================================================
# 平均池化 SHR（指令 I9，UOP 6）
# ==============================================================================
"""AVERAGE POOLING - SHR"""

# With this step, we average the added values.

# ----------------------
# Defining the SHR UOP buffer

if (len(uop_buffer) < 6 + 1):
    uop_buffer.append(structures_insn_uop.VTAUop( # UOP 6 - ALU (shift right)
        dst_idx=0, 
        src_idx=0,
        wgt_idx=0
    ))

# ----------------------
# Defining the ALU-SHR Instruction buffer

index_insn = 9 # Instruction index

if (len(insn_buffer) < index_insn + 1):
    insn_buffer.append(structures_insn_uop.VTAAluInsn( # I9: ALU - SHR (Average Pooling 3/3)
        opcode=4, # 4-ALU
        # DEP FLAG
        pop_prev_dep=0,
        pop_next_dep=0,
        push_prev_dep=0,
        push_next_dep=1, # Ready signal to STORE
        # Operations
        reset=0, # 0-no, 1-reset
        uop_bgn=6, # UOP 6
        uop_end=7,
        loop_out=14,
        loop_in=14,
        # UNUSED
        unused=0, # UNUSED
        # Index factors
        dst_factor_out=56,
        dst_factor_in=2, 
        src_factor_out=56,
        src_factor_in=2, 
        alu_opcode=3, # 0-MIN, 1-MAX, 2-ADD, 3-SHR, 4-MUL
        use_imm=1, # 0-no, 1-yes
        imm=2 # Division by 4 (rounded down)
    ))

# ----------------------
# Defining SHR operation

def SHR(A, IMM) :
    for i in range(len(A)): # A composed of horizontal vectors (16 x 1)
        A[i] = int(np.float64(A[i])) >> IMM
    return A

# ----------------------
# Pseudo-code ALU SHR

def insn_SHR(ACC):
    for i0 in range(insn_buffer[index_insn].loop_in):
        for i1 in range(insn_buffer[index_insn].loop_out):
            for uop_index in range(insn_buffer[index_insn].uop_bgn, insn_buffer[index_insn].uop_end):
                X = uop_buffer[uop_index].dst_idx
                dst_idx = i0 * insn_buffer[index_insn].dst_factor_in + i1 * insn_buffer[index_insn].dst_factor_out + X
                ACC[dst_idx] = SHR(ACC[dst_idx], insn_buffer[index_insn].imm)
    return ACC

# ----------------------
# Printing the data
# ----------------------

# Printing SHR UOP Buffer
print_uop_buffer("SHR", insn_buffer[index_insn].uop_bgn, insn_buffer[index_insn].uop_end)

# Printing SHR Instruction Buffer 
print_insn_buffer_ALU(index_insn, "SHR")

# Printing the Output Matrix
ACC_SHR = insn_SHR(ACC_ADD2)
print("\nACC - Output matrix post-SHR (", ACC_SHR.shape[0], "x", ACC_SHR.shape[1], ")")
print(ACC_SHR)

# ==============================================================================
# STORE / FINISH（指令 I10–I13）
# ==============================================================================
"""DATA STORAGE FROM SRAM TO DRAM"""

insn_buffer.append(structures_insn_uop.VTAMemInsn( # I10: STORE
    opcode=1, # 0-LOAD, 1-STORE, 3-FINISH
    # DEP FLAG
    pop_prev_dep=1, # Acknowledge COMPUTE ready signal
    pop_next_dep=0,
    push_prev_dep=1, # Ready signal to COMPUTE
    push_next_dep=0,
    # Memory interaction
    buffer_id=4, # 0-UOP, 1-WGT, 2-INP, 3-ACC, 4-OUT, 5-ACC8bit
    sram_base=0x0000,
    dram_base=0x00000300,
    unused=0, # UNUSED
    # Operation over the data
    y_size=1,
    x_size=784, # Store 49*16 OUT
    x_stride=784,
    y_pad_top=0,
    y_pad_bottom=0,
    x_pad_left=0,
    x_pad_right=0
))

insn_buffer.append(structures_insn_uop.VTAMemInsn( # I11: NOP-MEMORY-STAGE
    opcode=0, # 0-LOAD, 1-STORE, 3-FINISH
    # DEP FLAG
    pop_prev_dep=0,
    pop_next_dep=0,
    push_prev_dep=0, 
    push_next_dep=1, # Ready signal to COMPUTE
    # Memory interaction
    buffer_id=2, # 0-UOP, 1-WGT, 2-INP, 3-ACC, 4-OUT, 5-ACC8bit
    sram_base=0x0000,
    dram_base=0x00000000,
    unused=0, # UNUSED
    # Operation over the data
    y_size=0,
    x_size=0,
    x_stride=0,
    y_pad_top=0,
    y_pad_bottom=0,
    x_pad_left=0,
    x_pad_right=0
))

insn_buffer.append(structures_insn_uop.VTAMemInsn( # I12: NOP-COMPUTE-STAGE
    opcode=0, # 0-LOAD, 1-STORE, 3-FINISH
    # DEP FLAG
    pop_prev_dep=1, # Acknowledge LOAD ready signal
    pop_next_dep=1, # Acknowledge STORE ready signal
    push_prev_dep=0,
    push_next_dep=0,
    # Memory interaction
    buffer_id=0, # 0-UOP, 1-WGT, 2-INP, 3-ACC, 4-OUT, 5-ACC8bit
    sram_base=0x0000,
    dram_base=0x00000000,
    unused=0, # UNUSED
    # Operation over the data
    y_size=0,
    x_size=0,
    x_stride=0,
    y_pad_top=0,
    y_pad_bottom=0,
    x_pad_left=0,
    x_pad_right=0
))

insn_buffer.append(structures_insn_uop.VTAMemInsn( # I13: FINISH
    opcode=3, # 0-LOAD, 1-STORE, 3-FINISH
    # DEP FLAG
    pop_prev_dep=0,
    pop_next_dep=0,
    push_prev_dep=0,
    push_next_dep=0,
    # Memory interaction
    buffer_id=0, # 0-UOP, 1-WGT, 2-INP, 3-ACC, 4-OUT, 5-ACC8bit
    sram_base=0x0000,
    dram_base=0x00000000,
    unused=0, # UNUSED
    # Operation over the data
    y_size=0,
    x_size=0,
    x_stride=0,
    y_pad_top=0,
    y_pad_bottom=0,
    x_pad_left=0,
    x_pad_right=0
))

# ==============================================================================
# 二进制导出（对应 notebook %run insn_lenet5_layer1.py）
# ==============================================================================

def run_binary_generation():
    """运行 insn_lenet5_layer1.py，写出 uop.bin / instructions.bin。"""
    import runpy
    import structures as structures_insn_uop_mod

    def compiler_output_filepath(filename):
        out_dir = TUTORIAL_DIR.parent / "compiler_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / filename)

    structures_insn_uop_mod.compiler_output_filepath = compiler_output_filepath
    sys.modules["structures_insn_uop"] = structures_insn_uop_mod

    script = (
        TUTORIAL_DIR
        / "../src/compiler/vta_compiler/operations_definition/examples/insn_lenet5_layer1.py"
    )
    # insn 示例脚本依赖旧模块名中的辅助函数
    structures_insn_uop_mod.print_hex_128bit = lambda insn, debug=True: structures_insn_uop_mod.hex_128bit(insn, debug=debug)
    structures_insn_uop_mod.print_hex_32bit = lambda uop, debug=True: structures_insn_uop_mod.hex_32bit(uop, debug=debug)

    print(f"\n生成二进制指令: {script.resolve()}")
    runpy.run_path(str(script), run_name="__main__")
    print("已写入 compiler_output/uop.bin 与 instructions.bin")


if __name__ == "__main__":
    run_binary_generation()
