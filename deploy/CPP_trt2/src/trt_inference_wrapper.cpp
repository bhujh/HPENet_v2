#include "trt_inference_wrapper.h"
#include "pipeline.h"
#include "types.h"
#include "ply_reader.h"
#include <string>
#include <iostream>
#include <mutex>
#include <stdexcept>

// 使用 std::mutex 保护全局错误状态（线程安全考虑）
static std::mutex g_mutex;
static TensorrtInferencePipeline_C* g_trtinfer_handle = nullptr;
static TrLogger glogger;
// ── 不透明结构体定义 ─────────────────────────────────────────────────────
// 这个结构体只在 .cpp 文件中定义，对 C 侧完全透明
struct TensorrtInferencePipeline_C {
    std::unique_ptr<InferencePipeline> cpp_pipeline;
    mutable std::string last_error; // 存储错误信息

};

// ── Internal C++ helpers (需求中的两个独立函数) ──────────────────────────

namespace {

    /**
     * @brief [数据转换函数]
     * 从 cdi_t 数组提取 x,y,z,rcs,snr,v 到独立的特征数组。
     *
     * 对应需求: "添加一个数据转换函数。从 inst->pCdiPkg.cdis 中获取雷达点云的
     * x,y,z,rcs,snr,v 字段数据，转换成表示点云坐标和特征的结构体。"
     *
     * 通过 offset 参数访问 cdi_t 字段，避免在编译期依赖 rdp_types.h。
     */
    void convert_cdi_to_pointcloud(
        const void* cdis,
        int num_cdis,
        size_t elem_size,
        ptrdiff_t vcs_x_off,
        ptrdiff_t vcs_y_off,
        ptrdiff_t high_off,
        ptrdiff_t rcs_off,
        ptrdiff_t snr_off,
        ptrdiff_t v_off,
        float* coord_x,
        float* coord_y,
        float* coord_z,
        float* feat_rcs,
        float* feat_snr,
        float* feat_v
    ) {
        const char* base = static_cast<const char*>(cdis);
        for (int i = 0; i < num_cdis; i++) {
            const char* elem = base + static_cast<size_t>(i) * elem_size;
            coord_x[i] = *reinterpret_cast<const float*>(elem + vcs_x_off);
            coord_y[i] = *reinterpret_cast<const float*>(elem + vcs_y_off);
            coord_z[i] = *reinterpret_cast<const float*>(elem + high_off);
            feat_rcs[i] = *reinterpret_cast<const float*>(elem + rcs_off);
            feat_snr[i] = *reinterpret_cast<const float*>(elem + snr_off);
            feat_v[i] = *reinterpret_cast<const float*>(elem + v_off);
        }
    }

    /**
     * @brief [将cdi_t转为PointCloud]
     * 从 cdi_t 数组提取 x,y,z,rcs,snr,v 到独立的特征数组。
     *
     * 对应需求: "添加一个数据转换函数。从 inst->pCdiPkg.cdis 中获取雷达点云的
     * x,y,z,rcs,snr,v 字段数据，转换成表示点云坐标和特征的结构体。"
     *
     * 通过 offset 参数访问 cdi_t 字段，避免在编译期依赖 rdp_types.h。
     */
    void convert_cdi_to_pointcloud_v2(
        const void* cdis,
        int num_cdis,
        size_t elem_size,
        ptrdiff_t vcs_x_off,
        ptrdiff_t vcs_y_off,
        ptrdiff_t high_off,
        ptrdiff_t rcs_off,
        ptrdiff_t snr_off,
        ptrdiff_t v_off,
        ptrdiff_t valid_off,
        PointCloud& pc
    ) {
        const char* base = static_cast<const char*>(cdis);
        for (int i = 0; i < num_cdis; i++) {
            const char* elem = base + static_cast<size_t>(i) * elem_size;
            pc.label[i] = *reinterpret_cast<const float*>(elem + valid_off);
            pc.coord[3 * i + 0] = *reinterpret_cast<const float*>(elem + vcs_x_off);
            pc.coord[3 * i + 1] = *reinterpret_cast<const float*>(elem + vcs_y_off);
            pc.coord[3 * i + 2] = *reinterpret_cast<const float*>(elem + high_off);
            pc.feat[3 * i + 0] = *reinterpret_cast<const float*>(elem + rcs_off);
            pc.feat[3 * i + 1] = *reinterpret_cast<const float*>(elem + snr_off);
            pc.feat[3 * i + 2] = *reinterpret_cast<const float*>(elem + v_off);
        }
    }

    /**
     * @brief [结果更新函数]
     * 将 AI 推理的逐点预测结果写回 cdi_t.valid 字段。
     *
     * 对应需求: "添加一个结果更新函数。在推理结束后，使用结果
     * result.predictions 覆盖 inst->pCdiPkg.cdis 的 valid 的值。"
     */
    void update_predictions_to_cdi(
        void* cdis,
        int num_cdis,
        size_t elem_size,
        ptrdiff_t valid_off,
        const int* predictions
    ) {
        char* base = static_cast<char*>(cdis);
        for (int i = 0; i < num_cdis; i++) {
            char* elem = base + static_cast<size_t>(i) * elem_size;
            *reinterpret_cast<uint8_t*>(elem + valid_off) =
                static_cast<uint8_t>(predictions[i]);
        }
    }

} // namespace


// ── C API 实现 ───────────────────────────────────────────────────────────

extern "C" {

    //DEPLOYAI_LIB_API TensorrtInferencePipeline_C* trt_pipeline_create(
    //    const char* onnx_path,
    //    const char* stats_json_path
    //) {
    //    if (!onnx_path || !stats_json_path) return nullptr;
    //    // 允许重新初始化: 先销毁旧管线
    //    /*if (g_trtinfer_handle) {
    //        onnx_pipeline_destroy();
    //    }*/
    //    try {
    //        auto handle = std::make_unique<TensorrtInferencePipeline_C>();

    //        handle->cpp_pipeline = std::make_unique<InferencePipeline>(
    //            std::string(onnx_path),
    //            std::string(stats_json_path),
    //            logger
    //        );
    //        //释放所有权，将裸指针
    //        //g_trtinfer_handle = handle.release();
    //        return handle.release();
    //    }
    //    catch (const std::exception& e) {
    //        // 创建失败，返回 NULL
    //        std::cerr << "WARNING: ONNX initialization failed: " << e.what();
    //        return nullptr;
    //    }
    //}

    //DEPLOYAI_LIB_API void trt_pipeline_destroy(TensorrtInferencePipeline_C** g_trtinfer_handle) {
    //    if (!g_trtinfer_handle) return;          // 传入的指针本身为 NULL
    //    TensorrtInferencePipeline_C* p = *g_trtinfer_handle;
    //    if (!p) return;               // *handle 已经为 NULL
    //    {
    //        std::lock_guard<std::mutex> lock(g_mutex);
    //        delete p;                     // 释放对象
    //        *g_trtinfer_handle = nullptr;            // 将调用者的指针置空
    //    }
    //    
    //}


    DEPLOYAI_LIB_API TensorrtInferencePipeline_C* trt_pipeline_create(
        const char* onnx_path,
        const char* stats_json_path
    ) {
        if (!onnx_path || !stats_json_path) return nullptr;
        // 允许重新初始化: 先销毁旧管线
        if (g_trtinfer_handle) {
            trt_pipeline_destroy();
            // return nullptr;
        }
        try {
            auto handle = std::make_unique<TensorrtInferencePipeline_C>();

            handle->cpp_pipeline = std::make_unique<InferencePipeline>(
                std::string(onnx_path),
                std::string(stats_json_path),
                glogger
            );
            //释放所有权，将裸指针
            g_trtinfer_handle = handle.release();
            return g_trtinfer_handle;
        }
        catch (const std::exception& e) {
            // 创建失败，返回 NULL
            std::cerr << "ERROR: ONNX initialization failed: " << e.what();
            return nullptr;
        }
    }


    DEPLOYAI_LIB_API void trt_pipeline_destroy() {
        if (!g_trtinfer_handle) return;          // g_trtinfer_handle指针本身为 NULL
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            delete g_trtinfer_handle;                     // 释放对象
            g_trtinfer_handle = nullptr;            // 将调用者的指针置空
        }

    }


    DEPLOYAI_LIB_API float trt_pipeline_process_file(
        const char* ply_path,
        const char* output_path
    ) {
        if (!g_trtinfer_handle || !g_trtinfer_handle->cpp_pipeline || !ply_path || !output_path) {
            return -1.0f; // 参数错误
        }

        try {
            // 1. 重新加载点云数据（为了保留原始 PLY 的所有属性）
            PointCloud pc = PlyReader::loadv2(std::string(ply_path));
            if (pc.num_points == 0) {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_trtinfer_handle->last_error = "Loaded point cloud is empty: " + std::string(ply_path);
                return -2.0f;
            }

            // 2. 执行推理
            auto result = g_trtinfer_handle->cpp_pipeline->process_pointcloud(pc);

            // 3. 写入带标签的 PLY
            // 注意：这里我们复用了原有的 write_annotated_ply，
            // 但请注意原函数会覆盖文件且只写入固定字段。
            // 如果需要保留原始字段，建议扩展 write_annotated_ply 或在此处实现追加逻辑。
            write_annotated_ply<int>(std::string(output_path), pc, result.predictions);

            // 4. 返回延迟
            return result.latency_ms;

        }
        catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_trtinfer_handle->last_error = e.what();
            return -3.0f; // 推理/写入失败
        }
    }


    DEPLOYAI_LIB_API float trt_pipeline_process_inmemory(
        const float* coord_x,
        const float* coord_y,
        const float* coord_z,
        const float* feat_rcs,
        const float* feat_snr,
        const float* feat_v,
        int num_points,
        int* predictions_out
    ) {
        if (!g_trtinfer_handle || !g_trtinfer_handle->cpp_pipeline || !coord_x || !coord_y || !coord_z
            || !feat_rcs || !feat_snr || !feat_v || !predictions_out || num_points <= 0) {
            return -1.0f; // 参数错误
        }

        try {
            // 构造 PointCloud 结构
            PointCloud pc(num_points);
            for (int i = 0; i < num_points; i++) {
                pc.coord[i * 3 + 0] = coord_x[i];
                pc.coord[i * 3 + 1] = coord_y[i];
                pc.coord[i * 3 + 2] = coord_z[i];
                pc.feat[i * 3 + 0] = feat_rcs[i];
                pc.feat[i * 3 + 1] = feat_snr[i];
                pc.feat[i * 3 + 2] = feat_v[i];
            }

            // 执行推理
            auto result = g_trtinfer_handle->cpp_pipeline->process_pointcloud(pc);

            // 复制预测结果
            for (int i = 0; i < num_points; i++) {
                predictions_out[i] = result.predictions[i];
            }

            return result.latency_ms;

        }
        catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_trtinfer_handle->last_error = e.what();
            return -3.0f; // 推理失败
        }
    }


    DEPLOYAI_LIB_API float trt_ai_infer_and_update(
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
    ) {
        // 参数检查
        if (!g_trtinfer_handle)           return -1.0f;
        if (!cdis || num_cdis <= 0)       return -2.0f;

        try {
            // 1. 数据转换
            PointCloud pc(num_cdis);
            convert_cdi_to_pointcloud_v2(
                cdis, num_cdis, elem_size,
                vcs_x_off, vcs_y_off, high_off,
                rcs_off, snr_off, v_off, valid_off,
                pc
            );
            if (pc.num_points == 0) {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_trtinfer_handle->last_error = "Loaded point cloud is fail.";
                return -2.0f;
            }
            //保存初始雷达点云
            write_annotated_ply<float>(
                std::string("/home/wangpeng/CODE/HPENet_v2-main/deploy/"
                            "CPP_trt1/output/data0000071.ply"),
                pc, pc.label);
            std::cerr << "first write_annotated_ply" << std::endl;

            // 2. 执行推理
            if (!g_trtinfer_handle) {
                std::cerr << "handle is null" << std::endl;
                return -3.0f; // 推理/写入失败
            }
            auto result = g_trtinfer_handle->cpp_pipeline->process_pointcloud(pc);
            std::cerr << "inference" << std::endl;

            //保存预测结果雷达点云
            write_annotated_ply<int>(
                std::string("/home/wangpeng/CODE/HPENet_v2-main/deploy/"
                            "CPP_trt1/output/result0000071.ply"),
                pc, result.predictions);
            std::cerr << "second write_annotated_ply" << std::endl;

            // 3. 结果更新
            update_predictions_to_cdi(cdis, num_cdis, elem_size, valid_off, result.predictions.data());
            std::cerr << "update_predictions_to_cdi" << std::endl;

            // 4. 返回延迟
            return result.latency_ms;

        }
        catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_trtinfer_handle->last_error = e.what();
            return -3.0f; // 推理/写入失败
        }
    }


    DEPLOYAI_LIB_API float trt_ai_infer_all_radars(
        void* cdis_base,
        int total_cdis,
        size_t elem_size,
        ptrdiff_t vcs_x_off,
        ptrdiff_t vcs_y_off,
        ptrdiff_t high_off,
        ptrdiff_t rcs_off,
        ptrdiff_t snr_off,
        ptrdiff_t v_off,
        ptrdiff_t valid_off
    ) {
        // 参数检查
        if (!g_trtinfer_handle)           return -1.0f;
        if (!cdis_base || total_cdis <= 0) return -2.0f;

        try {
            // 1. 数据转换: 所有雷达 CDI → 单个 PointCloud
            PointCloud pc(total_cdis);
            convert_cdi_to_pointcloud_v2(
                cdis_base, total_cdis, elem_size,
                vcs_x_off, vcs_y_off, high_off,
                rcs_off, snr_off, v_off, valid_off,
                pc
            );

            // 2. 一次推理
            auto result = g_trtinfer_handle->cpp_pipeline->process_pointcloud(pc);

            // 3. 结果写回各 CDI 的 valid 字段
            update_predictions_to_cdi(
                cdis_base, total_cdis, elem_size,
                valid_off, result.predictions.data()
            );

            return result.latency_ms;

        }
        catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_trtinfer_handle->last_error = e.what();
            return -3.0f;
        }
    }


    DEPLOYAI_LIB_API const char* trt_get_last_error() {
        if (!g_trtinfer_handle) return "Invalid handle (null)";
        std::lock_guard<std::mutex> lock(g_mutex);
        return g_trtinfer_handle->last_error.c_str();
    }

} // extern "C"