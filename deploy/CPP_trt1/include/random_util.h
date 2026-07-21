#pragma once
#include <cstdint>

/**
 * @brief Numpy 兼容的 MT19937 伪随机数生成器
 *
 * 精确匹配 numpy.random.RandomState 的行为:
 * - 初始化: numpy 特有的 LCG 填充 624 状态字（与 C++ std::mt19937 不同）
 * - 生成: 标准 MT19937 twist + tempering
 * - uniform_int: 拒绝采样（匹配 numpy rk_interval）
 * - shuffle: Fisher-Yates
 *
 * 参考值 (seed=100):
 *   前 20 个 uniform_int(0, 9999):
 *   [5640, 6936, 8039, 79, 350, 4149, 7906, 5646, 802, 4376,
 *    9871, 1340, 7738, 9723, 5769, 6230, 2434, 5787, 1900, 4993]
 */
class NumpyMT19937 {
public:
    /// 构造并设置种子（默认 100，与 numpy.random.seed(100) 一致）
    explicit NumpyMT19937(uint32_t seed = 100);

    /// 生成 [low, high) 区间的均匀随机整数（匹配 numpy 的 rk_interval 拒绝采样）
    int uniform_int(int low, int high);

    /// 对 arr[0..n) 执行 Fisher-Yates 洗牌
    void shuffle(int* arr, int n);

    /// 生成 [0, 1) 的浮点数: genrand_uint32() / 2^32
    float uniform_float();

private:
    static const int N = 624;
    static const int M = 397;

    uint32_t mt[N];
    int mti;  // 当前状态索引, N 表示需要 twist

    /// numpy 特有初始化: LCG 填充状态数组
    void init_genrand(uint32_t seed);

    /// 生成 32 位随机数（twist + tempering）
    uint32_t genrand_uint32();
};
