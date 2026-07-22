// deployment_ai.h
#pragma once

// 跨平台导出宏定义
#if defined(DEPLOYAI_NO_EXPORTS)
#define DEPLOYAI_LIB_API 
#elif defined(_WIN32)
#ifdef DEPLOYAI_LIB_EXPORTS
#define DEPLOYAI_LIB_API __declspec(dllexport)
#else
#define DEPLOYAI_LIB_API __declspec(dllimport)
#endif
#elif defined(__linux__)
#define DEPLOYAI_LIB_API __attribute__((visibility("default")))
#endif
