"""
Tutorial 1 示例脚本：演示 LeNet-5 第一层卷积如何转换为 VTA 可用的矩阵形式。

================================================================================
一、卷积变 GEMM 是否等价？
================================================================================
等价。在相同的数值类型、相同的 padding/stride/dilation 设定下，
Im2Row（或 Im2Col）只是把「滑动窗口上的点积」重新组织成矩阵乘法，
不改变每个输出位置的计算公式。因此：

    卷积输出 Y  ⟺  矩阵乘积 ACC = A × B

其中 A 的每一行对应一个输出空间位置上的感受野（receptive field），
B 的每一列对应一个卷积核（filter）。


================================================================================
二、数学原理
================================================================================

2.1 二维卷积的定义
------------------
设输入张量 X 的形状为 (nc, nh, nw)：
  nc = 输入通道数, nh = 高, nw = 宽

设卷积核 W 的形状为 (mc, nc, fh, fw)：
  mc = 输出通道数（滤波器个数）, fh/fw = 核高/核宽

设 stride = (sh, sw)，padding 在高度/宽度方向分别为 (ph_top, ph_bottom)、(pw_left, pw_right)。

输出张量 Y 的形状为 (mc, mh, mw)，其中：

    mh = floor( (nh + ph_top + ph_bottom - fh) / sh ) + 1
    mw = floor( (nw + pw_left + pw_right - fw) / sw ) + 1

对固定的输出通道 m、空间位置 (i, j)，卷积定义为：

    Y[m, i, j] = Σ_{c=0}^{nc-1} Σ_{u=0}^{fh-1} Σ_{v=0}^{fw-1}
                   X[c, i·sh + u, j·sw + v] · W[m, c, u, v]

即：在位置 (i,j) 取一块大小为 nc×fh×fw 的感受野，与第 m 个核做逐元素乘再求和（点积）。


2.2 Im2Row 变换：把「滑动点积」写成矩阵乘法
-------------------------------------------
核心思想：每个输出位置 (i, j) 对应矩阵 A 的一行；每个滤波器 m 对应矩阵 B 的一列。

(1) 构造输入矩阵 A，形状 (mh·mw) × (nc·fh·fw)

    将输出平面上的位置 (i, j) 按行优先展平为行号：
        r = i · mw + j        （0 ≤ r < mh·mw）

    A 的第 r 行存放 Y[m, i, j] 计算时所需的感受野，按通道和核内坐标展开：

        A[r, k] = X[c, i·sh + u, j·sw + v]

    其中 k = c·(fh·fw) + u·fw + v  （0 ≤ k < nc·fh·fw）

    直观理解：卷积核在输入上每滑动一次，就把当前窗口内的 nc·fh·fw 个像素
    「拉直」成 A 的一行。因此 A 有 mh·mw 行（每个输出像素一行）。

(2) 构造权重矩阵 B，形状 (nc·fh·fw) × mc

    将第 m 个滤波器 W[m, :, :, :] 展平为 B 的一列：

        B[k, m] = W[m, c, u, v]

    其中 k 与上面相同的展平规则。B 有 mc 列（每个滤波器一列）。

(3) 矩阵乘法

        ACC = A × B        形状 (mh·mw) × mc

    对固定的 r 和 m：

        ACC[r, m] = Σ_k A[r, k] · B[k, m]

    展开后恰好就是 2.1 中的三重求和——与 Y[m, i, j] 完全相同的点积。
    因此 ACC[r, m] = Y[m, i, j]（在 (i,j) 与 r 的对应关系下）。

(4) 还原为输出张量

    将 ACC 的每一列 reshape 为 mh×mw，即得 mc 个输出特征图：

        Y[m, :, :] = reshape( ACC[:, m], (mh, mw) )


2.3 为何等价（本质原因）
------------------------
矩阵乘法 ACC[r, m] = A[r, :] · B[:, m] 是两个向量的内积。
- A[r, :] 是位置 (i,j) 的感受野向量
- B[:, m] 是第 m 个滤波器向量

这与卷积在 (i,j) 处对滤波器 m 所做的运算完全一致。
Im2Row 只是改变了数据的存储布局（layout），没有引入近似或省略项。


2.4 本脚本 LeNet-5 C1 的具体数值
--------------------------------
参数：nc=1, nh=nw=32, mc=6, fh=fw=5, sh=sw=1, padding=0

输出尺寸：
    mh = mw = (32 - 5) / 1 + 1 = 28

矩阵尺寸（与 tensor_matrix_converter.py 一致）：
    A (INP)  : (mh·mw) × (nc·fh·fw) = 784 × 25
    B (WGT)  : (nc·fh·fw) × mc       =  25 × 6
    ACC (OUT): (mh·mw) × mc          = 784 × 6

    784 = 28×28  （输出空间位置个数，即感受野个数）
     25 =  1×5×5  （每个感受野的像素数 = 输入通道 × 核面积）
      6 = 滤波器个数

举例：输出位置 (i=0, j=0)，r=0
    A[0, :] = 输入图像左上角 5×5 区域（1 通道）展平成的 25 维向量
    ACC[0, m] = A[0,:] · B[:,m] = 该 5×5 区域与第 m 个滤波器的卷积结果


2.5 与后续 padding / 分块的关系（重要区分）
-------------------------------------------
- Im2Row + GEMM：数学上等价于卷积，讨论的是 784×25 与 25×6 的乘法。
- 脚本中的 padding（784×25 → 784×32 等）：是为 VTA 硬件 16×16 分块对齐，
  在有效维度外补零，不改变有效区域内的乘法结果（零列/零行不参与有效点积）。
- 分块 GEMM：矩阵乘法的分块运算是线性可分的，分块累加结果与整体乘法一致。
- ReLU：在 GEMM 之后逐元素施加，不属于「卷积 ≡ GEMM」的等价范围，
  而是激活函数的后处理步骤。


================================================================================
三、脚本执行流程
================================================================================
1. 设置张量参数
   定义输入图像尺寸、卷积核尺寸、stride、padding 等。

2. 计算卷积输出张量尺寸
   调用 tensor_matrix_converter.output_dimension()
   32×32 输入 + 5×5 核 → 28×28 输出。

3. 计算 Im2Row 矩阵尺寸
   调用 tensor_matrix_converter.im2row_matrix_dimension()
   - 输入矩阵 A：784 行 × 25 列
   - 权重矩阵 B：25 行 × 6 列
   - 输出矩阵 ACC：784 行 × 6 列

4. 生成矩阵数据
   随机生成 A、B（或用全零矩阵），模拟编译器加载的原始数据。

5. 填充（padding，硬件对齐，非卷积数学本身所需）
   VTA 要求矩阵维度是 block_size(16) 的整数倍：
   - A：784×25 → 784×32
   - B：25×6   → 32×16

6. 分块（splitting）
   将填充后的矩阵切成 16×16 子块，供 VTA 逐块 GeMM：
   - A → 98 个 16×16 块
   - B → 2 个 16×16 块

7. 矩阵乘法验证
   a) 参考结果：numpy 对原始（未填充）A×B 做乘法 + ReLU
   b) 分块结果：按 VTA 分块方式逐块相乘累加 + ReLU

依赖模块
--------
- src/compiler/utils/tensor_matrix_converter.py  → 张量/矩阵维度换算
- src/compiler/vta_compiler/data_definition/data_definition.py  → 填充与分块
"""

import sys
from pathlib import Path

import numpy as np

# 基于脚本位置添加模块搜索路径
TUTORIAL_DIR = Path(__file__).resolve().parent
sys.path.append(str(TUTORIAL_DIR / "../src/compiler/utils"))
sys.path.append(str(TUTORIAL_DIR / "../src/compiler/vta_compiler/data_definition"))

import tensor_matrix_converter
from data_definition import matrix_creation, matrix_padding, matrix_splitting


def create_random_matrix(n_row, n_col, random_bound=4, dtype=np.int8):
    """生成随机矩阵（替代已移除的 matrix_generator.matrix_creation）。"""
    return np.random.randint(-random_bound, random_bound, size=(n_row, n_col), dtype=dtype)


def block_matrix_multiply(input_blocks, weight_blocks, input_block_col, weight_block_col, use_relu=False):
    """分块矩阵乘法（替代已移除的 matrix_multiplication）。"""
    output_blocks = []
    for i in range(len(input_blocks)):
        row = i // input_block_col
        col = i % input_block_col
        acc = np.zeros_like(input_blocks[i], dtype=np.int32)
        for k in range(weight_block_col):
            a_idx = row * weight_block_col + k
            b_idx = k * weight_block_col + col
            acc += np.matmul(input_blocks[a_idx].astype(np.int32), weight_blocks[b_idx].astype(np.int32))
        if use_relu:
            acc = np.maximum(acc, 0)
        output_blocks.append(acc.astype(np.int8))
    return output_blocks


def main():
    # --- 输入张量与卷积核（LeNet-5 C1）---
    input_channel = 1
    input_height = 32
    input_width = 32
    kernel_channel = 6
    kernel_height = 5
    kernel_width = 5
    stride_height = 1
    stride_width = 1
    pad_height = 0
    pad_width = 0

    # --- 计算 Im2Row 矩阵维度 ---
    output_tensor_height, output_tensor_width = tensor_matrix_converter.output_dimension(
        inp_dim=(input_height, input_width),
        wgt_dim=(kernel_height, kernel_width),
        stride=(stride_height, stride_width),
        padding=((pad_height, pad_height), (pad_width, pad_width)),
    )

    (inp_height, inp_width), (_, wgt_width), (out_height, out_width) = \
        tensor_matrix_converter.im2row_matrix_dimension(
            nc=input_channel, nh=input_height, nw=input_width,
            mc=kernel_channel, mh=output_tensor_height, mw=output_tensor_width,
            fh=kernel_height, fw=kernel_width,
            sh=stride_height, sw=stride_width,
            ph=(pad_height, pad_height), pw=(pad_width, pad_width),
            debug=True,
        )

    # --- 生成、填充、分块 ---
    block_size = 16
    is_init_random = True
    random_bound = 4

    if is_init_random:
        input_matrix = create_random_matrix(inp_height, inp_width, random_bound=random_bound)
        weight_matrix = create_random_matrix(inp_width, wgt_width, random_bound=random_bound)
    else:
        input_matrix = matrix_creation(m_row=inp_height, n_col=inp_width)
        weight_matrix = matrix_creation(m_row=inp_width, n_col=wgt_width)

    input_matrix_padded = matrix_padding(input_matrix, block_size=block_size, isWeight=False, isSquare=True)
    weight_matrix_padded = matrix_padding(weight_matrix, block_size=block_size, isWeight=True, isSquare=True)

    block_input_matrix, input_block_col, _ = matrix_splitting(
        input_matrix_padded, block_size, isWeight=False, isSquare=True,
    )
    block_weight_matrix, weight_block_col, _ = matrix_splitting(
        weight_matrix_padded, block_size, isWeight=True, isSquare=True,
    )

    print(f"\n输入矩阵 A: {input_matrix.shape} -> 填充后 {input_matrix_padded.shape} -> {len(block_input_matrix)} 个 {block_size}x{block_size} 块")
    print(f"权重矩阵 B: {weight_matrix.shape} -> 填充后 {weight_matrix_padded.shape} -> {len(block_weight_matrix)} 个 {block_size}x{block_size} 块")
    print(f"\nA0:\n{block_input_matrix[0]}\n")
    print(f"B0:\n{block_weight_matrix[0]}\n")

    # --- 参考矩阵乘法（未填充尺寸）---
    output_matrix = np.matmul(input_matrix.astype(np.int32), weight_matrix.astype(np.int32))
    output_matrix = np.maximum(output_matrix, 0)  # ReLU
    print(f"输出矩阵 ACC（参考）: {output_matrix.shape}, 前 3x3:\n{output_matrix[:3, :3]}\n")

    # --- 分块矩阵乘法 ---
    output_blocks = block_matrix_multiply(
        block_input_matrix, block_weight_matrix,
        input_block_col, weight_block_col, use_relu=True,
    )
    print(f"分块乘法完成: {len(output_blocks)} 个输出块, 首块形状 {output_blocks[0].shape}")


if __name__ == "__main__":
    main()
