/**
 * @file fsim_nn.cc
 * @brief 整网功能仿真（Functional Simulation）入口
 *
 * 本文件实现 fsim_nn 可执行程序的核心逻辑：按 NN 编译器生成的 dependency.csv
 * 调度表，模拟量化神经网络的前向推理。与 fsim_single_layer（单层独立验证）不同，
 * 本程序支持：
 *   - VTA 硬件层（GEMM/卷积等，经 VTADeviceRun 执行）
 *   - CPU 软件层（qadd、concat、quant、dequant、convtranspose）
 *   - 层间链式数据流（前层 int8 结果 res 作为后层输入）
 *   - im2row / int32 整形、padding、zero-point 与 rescaling
 *
 * 运行约定：工作目录为 functional_simulator/，通过相对路径
 * ../../../compiler_output/ 读取编译产物，最终写出 final_output.bin。
 *
 * 详见文档：docs/fsim_nn与fsim_single_layer_cn.md
 */

/***************************
    预处理器指令
****************************/
#include "../include/simulator_header.h"

// ---------------------------------------------------------------------------
// 数据类型别名（与 VTA 硬件位宽及量化路径一致）
// ---------------------------------------------------------------------------
using inp_dtype = int32_t;   // 输入矩阵 A / 输出 C 的元素类型
using wgt_dtype = int32_t;   // 权重矩阵 B
using acc_dtype = int32_t;   // 累加器 X、Y（int32 整形路径）

/**
 * @struct LayerContext
 * @brief 单层推理所需的全部上下文（主机缓冲 + VTA 虚拟 DRAM 指针）
 *
 * VTA 层在预加载阶段即分配虚拟 DRAM 并保持到整网结束；
 * CPU 层可能仅有 suffix/id，运行时动态填充 accX/accY/outC/res/value。
 */
struct LayerContext {
    int id;                    // 层序号（与 layers_name.csv 或 execution_order 索引相关）
    std::string suffix;        // 层名后缀，如 "QLinearConv1"，用作 layers_map 的键

    // --- 主机侧向量缓冲 ---
    std::vector<inp_dtype> inpA, outC;   // 输入 A（im2row 后）与 VTA/CPU 输出 C
    std::vector<wgt_dtype> wgtB;         // 权重
    std::vector<acc_dtype> accX, accY;   // 累加器 X/Y（int32 双输入路径）
    std::vector<uop_t> uop_buffer;       // VTA 微操作序列
    std::vector<instruction_t> insn_buffer;// VTA 指令流

    // --- 层间链式传递用的中间结果 ---
    std::vector<int8_t> res;   // rescaling 后的 int8 结果，供后续层作为输入
    std::vector<float> value;  // float 中间结果（dequant / convtranspose 路径）

    // --- VTA 虚拟 DRAM 指针（由 VTAMemAlloc 分配）---
    void* mem_inpA = nullptr;
    void* mem_wgtB = nullptr;
    void* mem_accX = nullptr;
    void* mem_accY = nullptr;
    void* mem_outC = nullptr;
    void* mem_uop  = nullptr;
    void* mem_insn = nullptr;

    vta_phy_addr_t phy_add_insn;  // 指令缓冲区的物理地址，供 VTADeviceRun 使用
};

/********************
    FSIM_NN 主函数
*********************/
/**
 * @brief 整网前向推理仿真
 * @return EXIT_SUCCESS 成功；EXIT_FAILURE 或 -1 表示失败
 *
 * 执行流程概览：
 *   0. 解析 compiler_output 路径
 *   1. 读取 layers_name.csv / dependency.csv，获取层数与 debug 标志
 *   2. 预加载所有 VTA 层（权重、累加器、uop、指令 → 虚拟 DRAM）
 *   3. 读取并格式化整网输入 input_nn.bin
 *   4. 初始化 TVM Profiler
 *   5. 按 dependency.csv 构建 execution_order 并逐层执行
 *   6. 释放 VTA 内存与设备
 *   7. 将最终层 res 写入 final_output.bin
 */
int fsim_nn() {
    // 是否在终端打印最终向量（默认 false，整网验证写 bin 即可）
    bool doPrint = false;

    // 当前工作目录（应为 functional_simulator/）
    std::filesystem::path currentPath = std::filesystem::current_path();

    // 构造指向仓库根目录下 compiler_output/ 中某文件的路径
    auto construct_path = [&](const std::string& filename) {
        return (currentPath / ".." / ".." / ".." / "compiler_output" / filename).string();
    };

    // =========================================================================
    // 0. 定义全局文件路径并加载 CSV 索引
    // =========================================================================
    // layers_name.csv：列出需预加载的 VTA IR 层（nb_vta_ir、后缀、debug）
    std::string fileLayerNamePath = construct_path("layers_name.csv");
    CsvMap layers_name_map = load_csv_to_map(fileLayerNamePath);

    // dependency.csv：整网拓扑、执行顺序、每层 processor/量化/形状/父层依赖
    std::string fileDependencyPath = construct_path("dependency.csv");
    CsvMap dependency_map = load_csv_to_map(fileDependencyPath);

    // 整网输入（int8，通常由 reference_onnx.py 生成）
    std::string fileInputNNPath = construct_path("input_nn.bin");

    // 整网最终输出（供 check_bin.py 与 reference.bin 比对）
    std::string fileFinalOutputPath = construct_path("final_output.bin");


    // =========================================================================
    // 1. 读取层数、VTA 层数与调试标志
    // =========================================================================
    // dependency.csv 中 nb_steps 行：整网执行步数（含 VTA + CPU 节点）
    int nb_steps = strToInt(get_csv_value(dependency_map, "nb_steps", 1));

    // layers_name.csv 中 nb_vta_ir 行：需要预加载二进制的 VTA 层数量
    int nb_vta_ir = strToInt(get_csv_value(layers_name_map, "nb_vta_ir", 1));

    // debug 标志位于 nb_vta_ir 行第 3 列（"True" 时打印调度与 profiler 信息）
    std::string debug_str = get_csv_value(layers_name_map, "nb_vta_ir", 2);
    bool debug = (debug_str == "True");

    if (debug) printf("\n\nThere are %d steps, %d are executed by the VTA! \n", nb_steps, nb_vta_ir);

    // 以层名（suffix）为键保存所有层的 LayerContext
    std::unordered_map<std::string, LayerContext> layers_map;
    layers_map.reserve(nb_steps);

    // 记录 VTA 层预加载顺序，用于结束时按序释放 VTAMem
    std::vector<std::string> loaded_layer_names;
    loaded_layer_names.reserve(nb_vta_ir);


    // =========================================================================
    // 2. 预加载并分配所有 VTA 层
    //    注意：此阶段不读 input{后缀}.bin，inpA 留空，运行时由前层输出链式填充
    // =========================================================================
    int block_size;  // VTA 块大小 BS，最后一层有效值会用于 output_tensor
    for (int i = 0; i < nb_vta_ir; ++i) {
        LayerContext ctx;
        ctx.id = i;

        // A. 从 layers_name.csv 获取当前 VTA 层的名称后缀
        ctx.suffix = get_csv_value(layers_name_map, std::to_string(i), 1);
        
        if (debug) printf("\n--- Loading Layer %d (Suffix: %s) ---\n", i, ctx.suffix.c_str());


        // B. 加载该层相关的 metadata 与二进制文件
        std::string fileMetadataPath = construct_path("metadata" + ctx.suffix + ".csv");
        CsvMap metadata_map = load_csv_to_map(fileMetadataPath);

        // 权重矩阵 B（VTA 块转置布局），由 main_vta_compiler 从 IR 中的 B 写出
        std::string fileWgtPath = construct_path("weight" + ctx.suffix + ".bin");
        // 累加器通道 X 初值（如卷积偏置），对应 metadata 中的矩阵 X / accX
        std::string fileAccPath = construct_path("accumulator" + ctx.suffix + ".bin");
        // 第二累加器通道 Y（双输入 ALU、残差加等），可为空；对应 metadata 中的矩阵 Y / accY
        std::string fileAddAccPath = construct_path("add_accumulator" + ctx.suffix + ".bin");
        // VTA 微操作表（UOP），指令流中 LOAD/STORE 等会按索引引用
        std::string fileUopPath = construct_path("uop" + ctx.suffix + ".bin");
        // VTA 硬件指令流，载入虚拟 DRAM 后 phy_add_insn 作为 VTADeviceRun 取指入口
        std::string fileInsnPath = construct_path("instructions" + ctx.suffix + ".bin");


        // C. 从 metadata.csv 读取块大小与各矩阵维度
        block_size = strToInt(get_csv_value(metadata_map, "BS", 1));

        // 矩阵 A（输入）：行、列、是否按 block 方阵 padding
        int A_row = strToInt(get_csv_value(metadata_map, "A", 1));
        int A_col = strToInt(get_csv_value(metadata_map, "A", 2));
        std::string A_square_str = get_csv_value(metadata_map, "A", 3);
        bool A_square = (A_square_str == "True");

        // 累加器 X
        int X_row = strToInt(get_csv_value(metadata_map, "X", 1));
        int X_col = strToInt(get_csv_value(metadata_map, "X", 2));
        std::string X_square_str = get_csv_value(metadata_map, "X", 3);
        bool X_square = (X_square_str == "True");

        // 累加器 Y（双输入 ALU 等）
        int Y_row = strToInt(get_csv_value(metadata_map, "Y", 1));
        int Y_col = strToInt(get_csv_value(metadata_map, "Y", 2));
        std::string Y_square_str = get_csv_value(metadata_map, "Y", 3);
        bool Y_square = (Y_square_str == "True");

        // 输出 C 缓冲空间
        int C_row = strToInt(get_csv_value(metadata_map, "C", 1));
        int C_col = strToInt(get_csv_value(metadata_map, "C", 2));
        std::string C_square_str = get_csv_value(metadata_map, "C", 3);
        bool C_square = (C_square_str == "True");

        
        // D. 读取数据并按 VTA 块布局做 data_formatting
        // 输入 A：预加载阶段留空（维度无效则空向量），运行时 im2row 再填充
        std::vector<inp_dtype> raw_inpA; 
        if (A_row <= 0 || A_col <= 0) ctx.inpA = raw_inpA;
        else ctx.inpA = data_formatting(raw_inpA, A_row, A_col, block_size, A_square);

        ctx.wgtB = read_binary_file<wgt_dtype>(fileWgtPath);

        std::vector<acc_dtype> raw_accX = read_binary_file<acc_dtype>(fileAccPath);
        if (X_row <= 0 || X_col <= 0) ctx.accX = raw_accX;
        else ctx.accX = data_formatting(raw_accX, X_row, X_col, block_size, X_square);

        std::vector<acc_dtype> raw_accY = read_binary_file<acc_dtype>(fileAddAccPath);
        if (Y_row <= 0 || Y_col <= 0) ctx.accY = raw_accY;
        else ctx.accY = data_formatting(raw_accY, Y_row, Y_col, block_size, Y_square);

        // 输出 C：预分配与格式化后的空缓冲，VTA 执行后写回
        std::vector<inp_dtype> raw_outC;
        if (C_row <= 0 || C_col <= 0) ctx.outC = raw_outC;
        else ctx.outC = data_formatting(raw_outC, C_row, C_col, block_size, C_square);

        ctx.uop_buffer = read_binary_file<uop_t>(fileUopPath);
        ctx.insn_buffer = read_binary_file<instruction_t>(fileInsnPath);


        // E. 在 VTA 虚拟 DRAM 中为各缓冲分配空间
        ctx.mem_inpA = VTAMemAlloc(ctx.inpA.size() * sizeof(inp_dtype), 1);
        ctx.mem_wgtB = VTAMemAlloc(ctx.wgtB.size() * sizeof(wgt_dtype), 1);
        ctx.mem_accX = VTAMemAlloc(ctx.accX.size() * sizeof(acc_dtype), 1);
        ctx.mem_accY = VTAMemAlloc(ctx.accY.size() * sizeof(acc_dtype), 1);
        ctx.mem_outC = VTAMemAlloc(ctx.outC.size() * sizeof(inp_dtype), 1);
        ctx.mem_uop  = VTAMemAlloc(ctx.uop_buffer.size() * sizeof(uop_t), 1);
        ctx.mem_insn = VTAMemAlloc(ctx.insn_buffer.size() * sizeof(instruction_t), 1);

        // 记录指令流物理地址，VTADeviceRun 从此地址取指
        ctx.phy_add_insn = VTAMemGetPhyAddr(ctx.mem_insn);
        

        // F. 将主机侧数据拷贝到虚拟 DRAM（inpA 可能为空，仍分配并写入）
        VTAMemCopyFromHost(ctx.mem_inpA, ctx.inpA.data(), ctx.inpA.size() * sizeof(inp_dtype));
        VTAMemCopyFromHost(ctx.mem_wgtB, ctx.wgtB.data(), ctx.wgtB.size() * sizeof(wgt_dtype));
        VTAMemCopyFromHost(ctx.mem_accX, ctx.accX.data(), ctx.accX.size() * sizeof(acc_dtype));
        VTAMemCopyFromHost(ctx.mem_accY, ctx.accY.data(), ctx.accY.size() * sizeof(acc_dtype));
        VTAMemCopyFromHost(ctx.mem_outC, ctx.outC.data(), ctx.outC.size() * sizeof(inp_dtype));
        VTAMemCopyFromHost(ctx.mem_uop,  ctx.uop_buffer.data(), ctx.uop_buffer.size() * sizeof(uop_t));
        VTAMemCopyFromHost(ctx.mem_insn, ctx.insn_buffer.data(), ctx.insn_buffer.size() * sizeof(instruction_t));


        // G. 存入 layers_map，并记录预加载顺序
        layers_map[ctx.suffix] = ctx;
        loaded_layer_names.push_back(ctx.suffix);
    }

    // H. 读取整网输入并按 image 段形状做块对齐格式化
    std::vector<int8_t> input_nn = read_binary_file<int8_t>(fileInputNNPath);

    int input_nn_height = strToInt(get_csv_value(dependency_map, "image", 1));
    int input_nn_width = strToInt(get_csv_value(dependency_map, "image", 2));

    // 首层依赖名为 "image" 时使用此向量；按 NCHW 方阵 padding
    input_nn = data_formatting(input_nn, input_nn_height, input_nn_width, block_size, true);


    // =========================================================================
    // 3. TVM Profiler 初始化（统计 VTA 指令执行）
    // =========================================================================
    const tvm::runtime::PackedFunc* profiler_clear = tvm::runtime::Registry::Get("vta.simulator.profiler_clear");
    const tvm::runtime::PackedFunc* profiler_status = tvm::runtime::Registry::Get("vta.simulator.profiler_status");
    const tvm::runtime::PackedFunc* profiler_debug_mode = tvm::runtime::Registry::Get("vta.simulator.profiler_debug_mode");

    if (!profiler_clear || !profiler_status || !profiler_debug_mode) {
        std::cerr << "ERROR: Profiler functions not found." << std::endl;
        return -1;
    }
    (*profiler_clear)();
    int debug_flag = 0; 
    (*profiler_debug_mode)(debug_flag);

    
    // =========================================================================
    // 4. 按 dependency.csv 构建整网执行顺序
    //    行 0..nb_steps-1 的第 2 列为层名；不在 layers_map 中的 CPU 层自动建空上下文
    // =========================================================================
    std::vector<std::string> execution_order;
    execution_order.reserve(nb_steps); 

    for (int i = 0; i < nb_steps; ++i) {
        std::string layer_to_execute = get_csv_value(dependency_map, std::to_string(i), 2);
       
        // CPU 层（qadd/concat 等）未出现在 layers_name.csv，需运行时创建占位 LayerContext
        if (layers_map.find(layer_to_execute) == layers_map.end()) {
            LayerContext ctx;
            
            ctx.id = i; 
            ctx.suffix = layer_to_execute;

            layers_map[layer_to_execute] = ctx;

            if (debug) printf("   -> Auto-created context for CPU layer: %s\n", layer_to_execute.c_str());
        }

        execution_order.push_back(layer_to_execute);
    }



    // =========================================================================
    // 5. 按 execution_order 逐层执行（VTA + CPU 混合）
    // =========================================================================
    VTADeviceHandle vta_device = VTADeviceAlloc();

    for (const std::string& layer_name : execution_order) {
        if (layers_map.find(layer_name) == layers_map.end()) {
            std::cerr << "ERROR: Layer " << layer_name << " not found!" << std::endl;
            continue; 
        }


        // A. 取当前层上下文
        LayerContext& ctx = layers_map[layer_name];

        if (debug) printf("\n--- Executing Layer: %s (ID: %d) --- \n", ctx.suffix.c_str(), ctx.id);


        // B. 从 dependency.csv 中以层名为键读取该层的调度元数据
        //    列索引约定见 docs/dependency_csv详解.md
        std::string processor = get_csv_value(dependency_map, ctx.suffix.c_str(), 1);   // vta / qadd / concat / ...
        std::string reshape_info = get_csv_value(dependency_map, ctx.suffix.c_str(), 2); // im2row / int32

        if (debug) printf("\t On %s with reshape %s \n", processor.c_str(), reshape_info.c_str());

        // Zero-point（量化零点偏移）
        int offsetA = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 3));
        int offsetB = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 5));
        int offsetU = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 7));
        int offsetV = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 9));
        int offsetC = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 25));

        // 各输入/输出的量化 scale
        float scaleA = strToFloat(get_csv_value(dependency_map, ctx.suffix.c_str(), 4));
        float scaleB = strToFloat(get_csv_value(dependency_map, ctx.suffix.c_str(), 6));
        float scaleU = strToFloat(get_csv_value(dependency_map, ctx.suffix.c_str(), 8));
        float scaleV = strToFloat(get_csv_value(dependency_map, ctx.suffix.c_str(), 10));
        float scaleC = strToFloat(get_csv_value(dependency_map, ctx.suffix.c_str(), 26));
        // VTA 层输出 rescaling 因子：Sa * Sb / Sc
        float scale = strToFloat(get_csv_value(dependency_map, ctx.suffix.c_str(), 27));

        // 输入张量 NCHW 形状
        int tensor_channel = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 11));
        int tensor_height = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 12));
        int tensor_width = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 13));

        // 输出张量 NCHW 形状
        int out_tensor_channel = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 22));
        int out_tensor_height = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 23));
        int out_tensor_width = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 24));

        // 卷积核高宽
        int kh = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 14));
        int kw = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 15));

        // 步长（当前仅使用 sh）
        int sh = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 16));
        // int sw = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 17));

        // 四向 padding：p0=Top, p1=Left, p2=Bottom, p3=Right
        int p0 = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 18));
        int p1 = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 19));
        int p2 = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 20));
        int p3 = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 21));

        // 父层依赖数量（1~4）
        int nb_inp = strToInt(get_csv_value(dependency_map, ctx.suffix.c_str(), 29));

        if (debug) printf("\t %d inputs: ", nb_inp);

        if (nb_inp < 1 || nb_inp > 4) {
            std::cerr << "ERROR: Too many inputs for layer " << ctx.suffix << std::endl;
            return EXIT_FAILURE;
        }

        // 父层名称（列 30 起）；首层输入依赖名为 "image"
        std::string name_dep, name_dep2, name_dep3, name_dep4;
        
        name_dep = get_csv_value(dependency_map, ctx.suffix.c_str(), 30);
        if (debug) printf("%s", name_dep.c_str());
        if (nb_inp == 2){
            name_dep2 = get_csv_value(dependency_map, ctx.suffix.c_str(), 31);
            if (debug) printf(", %s", name_dep2.c_str());
        } 
        else if (nb_inp == 3){
            name_dep2 = get_csv_value(dependency_map, ctx.suffix.c_str(), 31);
            if (debug) printf(", %s", name_dep2.c_str());
            name_dep3 = get_csv_value(dependency_map, ctx.suffix.c_str(), 32);
            if (debug) printf(", %s", name_dep3.c_str());
        } 
        else if (nb_inp == 4){
            name_dep2 = get_csv_value(dependency_map, ctx.suffix.c_str(), 31);
            if (debug) printf(", %s", name_dep2.c_str());
            name_dep3 = get_csv_value(dependency_map, ctx.suffix.c_str(), 32);
            if (debug) printf(", %s", name_dep3.c_str());
            name_dep4 = get_csv_value(dependency_map, ctx.suffix.c_str(), 33);
            if (debug) printf(", %s", name_dep4.c_str());
        } 
        if (debug) printf("\n");


        // C. 执行前数据重组：将前层 int8 输出链式写入当前层的 inpA 或 accX/accY
        if (reshape_info == "int32"){
            // int32 路径：用于 QLinearAdd 等，输入为 int32 累加器而非 im2row 后的 A

            if (nb_inp == 1){
                // 单输入：取父层 res 或整网 input_nn
                std::vector<int8_t> dep_out;
                if (name_dep == "image"){
                    dep_out = input_nn;
                }
                else{
                    LayerContext& dep_ctx = layers_map[name_dep];
                    dep_out = dep_ctx.res;
                }

                std::vector<acc_dtype> reshaped_dep = convert_vector_type<acc_dtype>(dep_out);

                // 减去 zero-point
                if (offsetA != 0){
                    reshaped_dep = subtract_offset(reshaped_dep, offsetA);
                }
                // 按需四向 padding，填充值 -128（int8 量化下界）
                if (p0 > 0 || p1 > 0 || p2 > 0 || p3 > 0) {
                    if (debug) printf("\t -> Applying int32 padding: T=%d, L=%d, B=%d, R=%d\n", p0, p1, p2, p3);

                    reshaped_dep = pad_matrix(
                        reshaped_dep,
                        tensor_channel,
                        tensor_height,
                        tensor_width,
                        block_size,
                        {p0, p1, p2, p3},
                        -128
                    );
                }

                ctx.accX = reshaped_dep;

                if (ctx.mem_accX != nullptr) VTAMemCopyFromHost(ctx.mem_accX, ctx.accX.data(), ctx.accX.size() * sizeof(acc_dtype));
            }

            else if (nb_inp == 2) {
                // 双输入：分别链式填充 accX 与 accY
                LayerContext& dep_ctx = layers_map[name_dep];
                std::vector<int8_t> dep_out = dep_ctx.res;

                LayerContext& dep2_ctx = layers_map[name_dep2];
                std::vector<int8_t> dep2_out = dep2_ctx.res;

                std::vector<acc_dtype> reshaped_dep1 = convert_vector_type<acc_dtype>(dep_out);
                std::vector<acc_dtype> reshaped_dep2 = convert_vector_type<acc_dtype>(dep2_out);

                if (offsetA != 0){
                    reshaped_dep1 = subtract_offset(reshaped_dep1, offsetA);
                }

                if (offsetB != 0){
                    reshaped_dep2 = subtract_offset(reshaped_dep2, offsetB);
                }

                if (p0 > 0 || p1 > 0 || p2 > 0 || p3 > 0) {
                    if (debug) printf("\t -> Applying int32 padding: T=%d, L=%d, B=%d, R=%d\n", p0, p1, p2, p3);

                    reshaped_dep1 = pad_matrix(
                        reshaped_dep1,
                        tensor_channel,
                        tensor_height,
                        tensor_width,
                        block_size,
                        {p0, p1, p2, p3},
                        -128
                    );

                    reshaped_dep2 = pad_matrix(
                        reshaped_dep2,
                        tensor_channel,
                        tensor_height,
                        tensor_width,
                        block_size,
                        {p0, p1, p2, p3},
                        -128
                    );
                }

                ctx.accX = reshaped_dep1;
                ctx.accY = reshaped_dep2;

                if (ctx.mem_accX != nullptr) VTAMemCopyFromHost(ctx.mem_accX, ctx.accX.data(), ctx.accX.size() * sizeof(acc_dtype));
                if (ctx.mem_accY != nullptr) VTAMemCopyFromHost(ctx.mem_accY, ctx.accY.data(), ctx.accY.size() * sizeof(acc_dtype));
            }
        }

        else if (reshape_info == "im2row"){
            // im2row 路径：卷积/GEMM 输入，将 NCHW int8 展开为 VTA 所需的 inpA 块布局

            std::vector<int8_t> dep_out;
            if (name_dep == "image"){
                dep_out = input_nn;
            }
            else{
                LayerContext& dep_ctx = layers_map[name_dep];
                dep_out = dep_ctx.res;
            }

            std::vector<inp_dtype> rescaled_dep = convert_vector_type<inp_dtype>(dep_out);

            // reshape：im2row + 块对齐 + offset 处理
            ctx.inpA  = reshape(
                rescaled_dep,
                block_size,
                1,                  // batch_size
                tensor_channel,
                tensor_height,
                tensor_width,
                {kh,kw},
                sh,
                {p0,p1,p2,p3},
                true,               // isSquare
                offsetA
            );
                
            VTAMemCopyFromHost(ctx.mem_inpA, ctx.inpA.data(), ctx.inpA.size() * sizeof(inp_dtype));
        }


        // D. 按 processor 分发：VTA 硬件仿真 或 CPU 量化算子
        if (processor == "vta"){
            // 调用 sim_driver 执行指令流，结果写入 mem_outC
            int flag = VTADeviceRun(vta_device, ctx.phy_add_insn, ctx.insn_buffer.size(), 0);
            
            if (flag != 0) {
                std::cerr << "ERROR: Execution failed at layer " << ctx.suffix << std::endl;
                VTADeviceFree(vta_device);
                return EXIT_FAILURE;
            }
        
            VTAMemCopyToHost(ctx.outC.data(), ctx.mem_outC, ctx.outC.size() * sizeof(inp_dtype));
        }
        else if (processor == "qadd") {
            // 量化加法：out = round((Sa*X + Sb*Y) / Sc)，在浮点域分解以保证数值语义
            if (debug) printf("\t -> Processing QAdd (CPU)\n");

            if (ctx.accX.size() != ctx.accY.size()) {
                std::cerr << "ERROR: QAdd input mismatch size (accX: " 
                          << ctx.accX.size() << ", accY: " << ctx.accY.size() << ")" << std::endl;
                return EXIT_FAILURE;
            }
            else if (scaleC == 0.0f) { 
                std::cerr << "ERROR: ScaleC is zero for QAdd layer" << std::endl;
                return EXIT_FAILURE;
            }

            ctx.outC.resize(ctx.accX.size());

            for (size_t k = 0; k < ctx.accX.size(); ++k) {
                float valX = (float)ctx.accX[k] * scaleA;
                float valY = (float)ctx.accY[k] * scaleB;
                float val = (valX + valY) / scaleC;
                ctx.outC[k] = (inp_dtype)std::nearbyint(val);
            }

            // QAdd 输出已含正确量化，后续 rescaling 使用 scale=1
            scale = 1.0;
        }
        else if (processor == "concat") {
            // 通道维 qlinear_concat：多路 int8 res 对齐 scale/zp 后拼接
            LayerContext& dep_ctx = layers_map[name_dep];
            LayerContext& dep2_ctx = layers_map[name_dep2];
            LayerContext& dep3_ctx = layers_map[name_dep3];
            LayerContext& dep4_ctx = layers_map[name_dep4];

            std::vector<acc_dtype> dep_out, dep2_out, dep3_out, dep4_out;

            std::vector<std::vector<acc_dtype>> concat_inp;
            std::vector<std::vector<int>> concat_shapes;
            std::vector<float> concat_scales;
            std::vector<int32_t> concat_zps;

            std::vector<int> shape = {1, tensor_channel, tensor_height, tensor_width};

            dep_out = convert_vector_type<acc_dtype>(dep_ctx.res);
            dep2_out = convert_vector_type<acc_dtype>(dep2_ctx.res);
            if (nb_inp == 2){
                concat_inp = {dep_out, dep2_out};
                concat_shapes = {shape, shape};
                concat_scales = {scaleA, scaleB};
                concat_zps = {offsetA, offsetB};
            }
            else if (nb_inp == 3){
                dep3_out = convert_vector_type<acc_dtype>(dep3_ctx.res);

                concat_inp = {dep_out, dep2_out, dep3_out};
                concat_shapes = {shape, shape, shape};
                concat_scales = {scaleA, scaleB, scaleU};
                concat_zps = {offsetA, offsetB, offsetU};
            }
            else if (nb_inp == 4){
                dep3_out = convert_vector_type<acc_dtype>(dep3_ctx.res);
                dep4_out = convert_vector_type<acc_dtype>(dep4_ctx.res);

                concat_inp = {dep_out, dep2_out, dep3_out, dep4_out};
                concat_shapes = {shape, shape, shape, shape};
                concat_scales = {scaleA, scaleB, scaleU, scaleV};
                concat_zps = {offsetA, offsetB, offsetU, offsetV};
            }

            ctx.outC = qlinear_concat<acc_dtype>(
                concat_inp,
                concat_shapes,
                concat_scales,
                concat_zps,
                scaleC,
                offsetC,
                1,          // axis：通道维
                block_size
            );

            scale = 1.0;
            offsetC = 0;
        }
        else if (processor == "dequant") {
            // DequantizeLinear：int8 → float，结果存 ctx.value（不走 rescaling → res）
            if (debug) printf("\t -> Processing DequantizeLinear (CPU)\n");

            std::vector<int8_t> dep_out;
            if (name_dep == "image"){
                dep_out = input_nn;
            }
            else{
                LayerContext& dep_ctx = layers_map[name_dep];
                dep_out = dep_ctx.res;
            }

            std::vector<inp_dtype> rescaled_dep = convert_vector_type<inp_dtype>(dep_out);
            
            std::vector<float> float_res = dequantize_linear(rescaled_dep, scaleA, offsetA);

            ctx.value = float_res;
            
        }
        else if (processor == "quant") {
            // QuantizeLinear：float → int32 outC
            if (debug) printf("\t -> Processing QuantizeLinear (CPU)\n");

            LayerContext& dep_ctx = layers_map[name_dep];

            std::vector<float> float_res = dep_ctx.value;
            
            std::vector<acc_dtype> int_res = quantize_linear(float_res, scaleA, offsetA);

            ctx.outC = int_res;

            scale = 1.0;
            offsetC = 0;
            
        }
        else if (processor == "convtranspose") {
            // ConvTranspose：在 float 域执行，读 float 权重/偏置 bin
            if (debug) printf("\t -> Processing ConvTranspose (CPU)\n");

            std::string fileCTWgtPath = construct_path("weight" + ctx.suffix + ".bin");
            std::string fileCTAccPath = construct_path("accumulator" + ctx.suffix + ".bin");

            std::vector<float> CTwgt = read_binary_file<float>(fileCTWgtPath);
            std::vector<float> CTacc = read_binary_file<float>(fileCTAccPath);

            LayerContext& dep_ctx = layers_map[name_dep];

            std::vector<float> float_dep_out = dep_ctx.value;
            
            // 向量 → 重建 NCHW 张量 → 反卷积 → 展平为 float 向量
            std::vector<float> float_res = conv_transpose(
                float_dep_out, 
                CTwgt, CTacc, 
                1, 
                tensor_channel, tensor_height, tensor_width,
                out_tensor_channel, out_tensor_height, out_tensor_width,
                kh, kw, sh, 
                {p0, p1, p2, p3}, 
                block_size
            );

            ctx.value = float_res;
            
        }
        else { 
            // 未知 processor：占位（无操作）
            NULL;
        }


        // E. 将 outC 量化为 int8 链式结果 res（dequant/convtranspose 跳过，它们使用 ctx.value）
        if ((processor != "dequant") && (processor != "convtranspose")){
            if (debug) printf("\nRescaling: \n\t rescaling factor=%.18lf and offset=%d \n", scale, offsetC);

            ctx.res = rescaling(
                ctx.outC,
                scale,
                offsetC
            );
        }
    }


    // =========================================================================
    // 6. 释放资源并可选输出 Profiler 报告
    // =========================================================================
    if (debug) {
        std::string profile_json = (*profiler_status)();
        std::cout << "\n--- Profiler Status ---" << std::endl << profile_json << std::endl;
    }

    // 仅释放预加载过的 VTA 层虚拟 DRAM（CPU 层无 mem_* 分配）
    for (const std::string& layer_name : loaded_layer_names) {
        LayerContext& ctx = layers_map[layer_name]; 

        VTAMemFree(ctx.mem_inpA);
        VTAMemFree(ctx.mem_wgtB);
        VTAMemFree(ctx.mem_accX);
        VTAMemFree(ctx.mem_accY);
        VTAMemFree(ctx.mem_outC);
        VTAMemFree(ctx.mem_uop);
        VTAMemFree(ctx.mem_insn);
    }

    VTADeviceFree(vta_device);


    // =========================================================================
    // 7. 写出最终输出 final_output.bin
    //    从 dependency.csv 的 output 段读取最终层名与 NCHW 形状
    // =========================================================================
    std::string output_name = get_csv_value(dependency_map, "output", 1);
    int tensor_channel = strToInt(get_csv_value(dependency_map, "output", 2));
    int tensor_height = strToInt(get_csv_value(dependency_map, "output", 3));
    int tensor_width = strToInt(get_csv_value(dependency_map, "output", 4));

    LayerContext& ctx = layers_map[output_name];

    if (doPrint) {
        printf("\n\nRESULT LAYER %d:\n", ctx.id);
        printf("Final result = {");
        print_vector(ctx.res.data(), ctx.res.size());
        printf("\n} \n");
    }

    // 将 int8 向量按 NCHW 与 block 布局写回二进制，供 check_bin.py 校验
    output_tensor(
        ctx.res,
        block_size,
        1,
        tensor_channel,
        tensor_height,
        tensor_width,
        fileFinalOutputPath
    );

    return EXIT_SUCCESS;
}

/****************
    程序入口
*****************/
int main() {
    return fsim_nn();
}
