# 协变返回类型重构：消除 IPluginV2 deprecated 警告

**日期**: 2026-08-20 | **对象**: `deploy/trt_plugins/` 全部 9 个插件 creator

## 背景

构建 `libhpenet_plugins.so` 时出现 18 处 `warning: 'IPluginV2' is deprecated`：
- `IPluginCreator::createPlugin/deserializePlugin` 接口签名强制返回 `IPluginV2*`（NVIDIA 未升级）
- TRT 8.6.1.6 中 `IPluginV2` 与 `IPluginV2Ext` 均被 `TRT_DEPRECATED`（`NvInferRuntimePlugin.h:97/407`）

## 方案（协变返回类型）

C++ 协变返回类型允许 override 返回基类返回类型的派生类指针。将 creator 的
`createPlugin/deserializePlugin` 返回类型从 `IPluginV2*` 改为 `IPluginV2DynamicExt*`
（继承链 `IPluginV2DynamicExt → IPluginV2Ext → IPluginV2`，8.6.1.6 未弃用，
`NvInferRuntime.h:350`），且是插件类实际继承的类型，`return new XxxPlugin(...)` 无需转换。

- **否决的替代**：`IPluginV2Ext*` —— 实测 8.6.1.6 中 IPluginV2Ext 同样弃用，改了照样警告
- **功能等价**：TRT 经 `IPluginCreator` 基类指针调用，静态类型仍 IPluginV2*；协变返回是纯编译期特性，不影响 vtable/ABI/运行时行为

## 改动

17 文件 36 处（9 header 声明 18 处 + 8 cpp 实现 18 处，ballquerygroup 含双 creator）：

```
include/{ballquery,ballquerygroup,ballquerydp,fps,flashfps,fpsprune,prefixfps,samplefps,threeinterp}_plugin.h
src/{ballquery,ballquerygroup,fps,flashfps,fpsprune,prefixfps,samplefps,threeinterp}_plugin.cpp
```

统一 `IPluginV2*` → `IPluginV2DynamicExt*`（header 的 `nvinfer1::IPluginV2*` 子串一并命中；
`IPluginV2Ext*`/`IPluginV2DynamicExt*` 不含 `IPluginV2*` 子串，无误伤）。`plugin_registry.cpp`
用 `REGISTER_TENSORRT_PLUGIN` 宏（经基类接口调用），无需改。

## 验证

| 验证项 | 结果 |
|---|---|
| 干净重编（make clean && make） | **exit 0，0 警告**（deprecated 警告全消除） |
| 残留检查（grep `IPluginV2*` 非 Ext 后缀） | 0 处 |
| 9 个 creator 注册（getPluginCreator） | **9/9 OK** |
| FPSPrune 端到端（build engine + 推理 vs 改前 golden） | **ALL PASS**（B=1 bit 级 + B=2/3/8 逐 batch） |
| 现役 FPS 端到端对拍（tests_fps_algos.py vs openpoints FPS） | **100.0% / 100.0% 逐索引一致** |

## 结论

协变返回类型 `IPluginV2DynamicExt*` 精确消除 deprecated 警告，功能零变化（回归全过）。
纯编译期重构，不涉及插件语义/序列化/注册逻辑。
