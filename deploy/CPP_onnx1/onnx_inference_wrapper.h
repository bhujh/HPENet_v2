#
/**
 * @file onnx_c_api.h
 * @brief ONNX 推理管线的纯 C 语言接口。
 *
 * 此文件定义了通过不透明指针（Opaque Pointer）暴露给 C 语言的 API。
 * C 语言调用者只需包含此文件即可，无需了解底层的 C++ 实现。
 */

#ifndef ONNX_C_API_H
#define ONNX_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

    /*
     * 动态库导出宏定义 (可根据您的构建系统自行调整)
     * 如果您使用 CMake，通常通过 generate_export_header 自动生成；
     * 这里提供一个通用的占位宏。
     */
     // 跨平台导出宏定义
#ifdef _WIN32
#ifdef DEPLOYAI_LIB_EXPORTS
#define DEPLOYAI_LIB_API __declspec(dllexport)
#else
#define DEPLOYAI_LIB_API __declspec(dllimport)
#endif
#else
    // Linux/macOS 下默认全局符号可见，或显式指定
#define DEPLOYAI_LIB_API __attribute__((visibility("default")))
#endif


     /**
      * @brief 不透明句柄类型。
      *
      * 内部封装了 C++ 的 OnnxInferencePipeline 对象。
      * C 语言侧仅通过指针传递，不可直接访问或修改其内部结构。
      */
    typedef struct OnnxInferencePipeline_C OnnxInferencePipeline_C;

    /**
     * @brief 创建 ONNX 推理管线实例。
     *
     * @param onnx_path      ONNX 模型文件的路径 (UTF-8 字符串)。
     * @param stats_json_path 统计数据 JSON 文件的路径 (UTF-8 字符串)。
     * @param min_n          最小点数阈值。
     * @param max_n          最大点数阈值。
     * @param voxel_size     体素大小。
     * @param seed           随机种子。
     * @return OnnxInferencePipeline_C* 成功返回有效的句柄指针；
     *         失败（如模型加载错误、内存分配失败）返回 NULL。
     */
    DEPLOYAI_LIB_API OnnxInferencePipeline_C* onnx_pipeline_create(
        const char* onnx_path,
        const char* stats_json_path
    );

    /**
     * @brief 销毁推理管线实例并释放相关内存。
     *
     * @param handle 指向 OnnxInferencePipeline_C 的指针。
     *               如果传入 NULL，函数将安全地直接返回。
     */
    DEPLOYAI_LIB_API void onnx_pipeline_destroy(void);

    /**
     * @brief 对指定的 PLY 文件执行推理并生成带标签的输出文件。
     *
     * @param handle      有效的推理管线句柄。
     * @param ply_path    输入 PLY 文件的路径。
     * @param output_path 输出带标签 PLY 文件的路径。
     * @return float 推理耗时（单位：毫秒）。
     *         如果发生错误，返回负数错误码：
     *         -1.0f: 参数无效或句柄为空
     *         -2.0f: 点云加载失败或为空
     *         -3.0f: 推理过程或文件写入发生异常
     */
    DEPLOYAI_LIB_API float onnx_pipeline_process_file(
        const char* ply_path,
        const char* output_path
    );

    /**
     * @brief 获取管线最后一次操作的错误信息。
     *
     * 当 onnx_pipeline_create 返回 NULL，或 onnx_pipeline_process_file
     * 返回负数时，可调用此函数获取具体的错误描述。
     *
     * @param handle 推理管线句柄。
     * @return const char* 错误信息的 UTF-8 字符串指针。
     *         如果没有错误，返回 NULL。
     *         注意：返回的字符串由内部内存管理，请勿手动 free。
     */
    DEPLOYAI_LIB_API const char* onnx_get_last_error(void);


    /**
     * @brief 对内存中的点云数据直接执行推理（不经过 PLY 文件读写）。
     *
     * @param handle       有效的推理管线句柄。
     * @param coord_x      点云 X 坐标数组（长度 n）。
     * @param coord_y      点云 Y 坐标数组（长度 n）。
     * @param coord_z      点云 Z 坐标数组（长度 n）。
     * @param feat_rcs     点云 RCS 特征数组（长度 n）。
     * @param feat_snr     点云 SNR 特征数组（长度 n）。
     * @param feat_v       点云速度特征数组（长度 n）。
     * @param num_points   点云数量。
     * @param predictions_out 输出：每个点的推理结果（长度 n，调用者分配内存）。
     * @return float 推理耗时（毫秒）。负数表示错误：
     *         -1.0f: 参数无效或句柄为空
     *         -2.0f: 点云为空或点数不足
     *         -3.0f: 推理过程发生异常
     */
    DEPLOYAI_LIB_API float onnx_pipeline_process_inmemory(
        const float* coord_x,
        const float* coord_y,
        const float* coord_z,
        const float* feat_rcs,
        const float* feat_snr,
        const float* feat_v,
        int num_points,
        int* predictions_out
    );


    
    DEPLOYAI_LIB_API float rdp_ai_infer_and_update(
        void* cdis,
        int num_cdis,
        size_t elem_size,
        ptrdiff_t vcs_x_off,
        ptrdiff_t vcs_y_off,
        ptrdiff_t high_off,
        ptrdiff_t rcs_off,
        ptrdiff_t snr_off,
        ptrdiff_t v_off,
        ptrdiff_t valid_off
    );


#ifdef __cplusplus
}
#endif

#endif /* ONNX_C_API_H */