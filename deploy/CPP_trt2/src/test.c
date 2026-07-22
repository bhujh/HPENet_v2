#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// 包含修改后的头文件
#include "trt_inference_wrapper.h"
//#include "onnx_inference.h"


//"D:/ProgramData/hpenet_deploy/trt_model_fp32.engine" "D:/ProgramData/hpenet_deploy/stats.json" "D:/ProgramData/hpenet_deploy/radarfull/raw/0000071.ply" "D:/ProgramData/hpenet_deploy/radarfull/output/0000071.ply"

#ifdef _WIN32
#include <windows.h>
#define PATH_SEP '\\'
#define MAX_PATH_LEN MAX_PATH
#elif defined(__linux__) || defined(__APPLE__)
#include <dirent.h>
#define PATH_SEP '/'
#define MAX_PATH_LEN 4096
#else
#error "不支持的操作系统平台"
#endif

/**
 * @brief 跨平台获取目录下的文件名及完整路径（非递归）
 * @param dir_path    存放数据的文件夹路径
 */
void list_files(const char* dir_path, const char* result_path) {
    if (dir_path == NULL || dir_path[0] == '\0') {
        fprintf(stderr, "错误: 目录路径不能为空\n");
        return;
    }

    char full_path[MAX_PATH_LEN];
    char output_file[MAX_PATH_LEN];
    size_t dir_len = strlen(dir_path);

    // 检查目录路径是否以分隔符结尾，避免拼接出双斜杠
    int has_trailing_sep = (dir_len > 0) && (dir_path[dir_len - 1] == '/' || dir_path[dir_len - 1] == '\\');

#ifdef _WIN32
    // --- Windows 平台实现 ---
    char search_path[MAX_PATH_LEN];
    // 安全拼接搜索路径，防止缓冲区溢出
    if (has_trailing_sep) {
        snprintf(search_path, MAX_PATH_LEN, "%s*", dir_path);
    }
    else {
        snprintf(search_path, MAX_PATH_LEN, "%s\\*", dir_path);
    }

    WIN32_FIND_DATAA find_data;
    HANDLE h_find = FindFirstFileA(search_path, &find_data);

    if (h_find == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "无法打开目录: %s (错误码: %lu)\n", dir_path, GetLastError());
        return;
    }

    do {
        if (strcmp(find_data.cFileName, ".") != 0 && strcmp(find_data.cFileName, "..") != 0) {

            //snprintf(full_path, MAX_PATH_LEN, "%s%c%s", dir_path, has_trailing_sep ? '\0' : PATH_SEP, find_data.cFileName);
            // 如果原路径有分隔符，直接拼接文件名即可
            if (has_trailing_sep) {
                snprintf(full_path, MAX_PATH_LEN, "%s%s", dir_path, find_data.cFileName);
                snprintf(output_file, MAX_PATH_LEN, "%sresult%s", result_path, find_data.cFileName);
            }
            else {
                snprintf(full_path, MAX_PATH_LEN, "%s%c%s", dir_path, PATH_SEP, find_data.cFileName);
                snprintf(output_file, MAX_PATH_LEN, "%s%cresult%s", result_path, PATH_SEP, find_data.cFileName);
            }
            printf("文件路径: %s\n", full_path);
            float latency = trt_pipeline_process_file(full_path, output_file);

            if (latency < 0) {
                // 获取具体的错误信息
                const char* err = trt_get_last_error();
                printf("Error: Inference failed. Code: %f, Details: %s\n", latency, err);
            }
            else {
                printf("Success! Inference latency: %.2f ms\n", latency);
                printf("Labeled result saved to: %s\n", output_file);
            }
        }
    } while (FindNextFileA(h_find, &find_data) != 0);

    FindClose(h_find);

#elif defined(__linux__) || defined(__APPLE__)
    // --- Linux / macOS 平台实现 ---
    DIR* dir = opendir(dir_path);
    if (dir == NULL) {
        perror("无法打开目录");
        return;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        // 同样处理尾部斜杠
        if (has_trailing_sep) {
            snprintf(full_path, MAX_PATH_LEN, "%s%s", dir_path, entry->d_name);
            snprintf(output_file, MAX_PATH_LEN, "%sresult%s", result_path, entry->d_name);
        }
        else {
            snprintf(full_path, MAX_PATH_LEN, "%s/%s", dir_path, entry->d_name);
            snprintf(output_file, MAX_PATH_LEN, "%s/result%s", result_path, entry->d_name);
        }
        printf("文件路径: %s\n", full_path);
        float latency = trt_pipeline_process_file(full_path, output_file);

        if (latency < 0) {
          // 获取具体的错误信息
          const char *err = trt_get_last_error();
          printf("Error: Inference failed. Code: %f, Details: %s\n", latency,
                 err);
        } else {
          printf("Success! Inference latency: %.2f ms\n", latency);
          printf("Labeled result saved to: %s\n", output_file);
        }
    }
    closedir(dir);
#endif
}



int main() {
    // 1. 创建推理器
    // 参数: model.onnx, stats.json, min_n, max_n, voxel_size, seed
    TensorrtInferencePipeline_C *pipe = trt_pipeline_create(
        "/home/wangpeng/CODE/HPENet_v2-main/deploy/trt_model_fp32.engine",
        "/home/wangpeng/CODE/HPENet_v2-main/deploy/CPP_trt/stats.json");

    if (!pipe) {
        printf("Error: Failed to create pipeline. Check model paths.\n");
        return -1;
    }

    // 2. 处理文件
    const char *input_file = "/home/wangpeng/CODE/HPENet_v2-main/data/"
                             "RadarClassi/radarfull/raw/0000071.ply";
    const char *output_file = "/home/wangpeng/CODE/HPENet_v2-main/deploy/CPP_trt1/output/baocun0000071.ply";
    const char *dir_path =
        "/home/wangpeng/CODE/HPENet_v2-main/data/RadarClassi/radarfull/raw/";
    const char *result_path =
        "/home/wangpeng/CODE/HPENet_v2-main/deploy/CPP_trt1/output/";

    list_files(dir_path, result_path);
    //printf("Processing %s ...\n", input_file);
    //float latency = trt_pipeline_process_file(input_file, output_file);

    //if (latency < 0) {
    //    // 获取具体的错误信息
    //    const char* err = trt_get_last_error();
    //    printf("Error: Inference failed. Code: %f, Details: %s\n", latency, err);
    //}
    //else {
    //    printf("Success! Inference latency: %.2f ms\n", latency);
    //    printf("Labeled result saved to: %s\n", output_file);
    //}

    // 3. 释放资源
    trt_pipeline_destroy();
    printf("Resources cleaned up. Exiting.\n");

    return 0;
}