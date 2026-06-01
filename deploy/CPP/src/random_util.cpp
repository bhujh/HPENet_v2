#include "random_util.h"

// ============================================================================
// numpy 特有 MT19937 初始化
//
// 与 C++ std::mt19937 不同:
//   std::mt19937 使用: state[i] = 1812433253 * (state[i-1] ^ (state[i-1] >> 30)) + i
//   但初始化和位操作细节有差异。
//
// numpy 的 RandomState 初始化 (randomkit.c:rk_seed):
//   state[0] = seed & 0xFFFFFFFF
//   for i in range(1, 624):
//       state[i] = 1812433253 * (state[i-1] ^ (state[i-1] >> 30)) + i
//       state[i] &= 0xFFFFFFFF
// ============================================================================
void NumpyMT19937::init_genrand(uint32_t seed) {
    mt[0] = seed & 0xFFFFFFFFUL;
    for (int i = 1; i < N; i++) {
        mt[i] = 1812433253UL * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i;
        mt[i] &= 0xFFFFFFFFUL;
    }
    mti = N;  // 标记需要首次 twist
}

NumpyMT19937::NumpyMT19937(uint32_t seed) : mti(N + 1) {
    init_genrand(seed);
}

// ============================================================================
// 标准 MT19937 生成器: twist + tempering
//
// 完全匹配 松本眞・西村拓土 原始论文 (Mersenne Twister, 1997)
// 以及 numpy randomkit.c 的实现
// ============================================================================
uint32_t NumpyMT19937::genrand_uint32() {
    const uint32_t LOWER_MASK = 0x7FFFFFFFUL;  // 低 31 位
    const uint32_t UPPER_MASK = 0x80000000UL;  // 高 1 位
    const uint32_t MAGIC      = 0x9908B0DFUL;  // 捻转常数

    // ---- Twist ----
    if (mti >= N) {
        int i;
        uint32_t y;

        // 第一段: k = 0 .. N-M-1 (0 .. 226)
        for (i = 0; i < N - M; i++) {
            y = (mt[i] & UPPER_MASK) | (mt[i + 1] & LOWER_MASK);
            mt[i] = mt[i + M] ^ (y >> 1) ^ (MAGIC & -(y & 1));
        }

        // 第二段: k = N-M .. N-2 (227 .. 622)
        for (; i < N - 1; i++) {
            y = (mt[i] & UPPER_MASK) | (mt[i + 1] & LOWER_MASK);
            mt[i] = mt[i + (M - N)] ^ (y >> 1) ^ (MAGIC & -(y & 1));
        }

        // 第三段: k = N-1 (623), 回绕到 mt[0]
        y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ (MAGIC & -(y & 1));

        mti = 0;
    }

    // ---- Tempering ----
    uint32_t y = mt[mti++];

    y ^= (y >> 11);
    y ^= (y << 7) & 0x9D2C5680UL;
    y ^= (y << 15) & 0xEFC60000UL;
    y ^= (y >> 18);

    return y;
}

// ============================================================================
// 生成 [low, high) 区间的均匀随机整数
//
// 精确匹配 numpy 的 rk_interval (randomkit.c):
//   1. 计算位掩码: 大于等于 (high-low) 的最小 2^k - 1
//   2. 拒绝采样: 生成 (value & mask) 直到 value <= (high-low)
//
// 注意: 使用拒绝采样而非取模，以完全匹配 numpy 输出。
// ============================================================================
int NumpyMT19937::uniform_int(int low, int high) {
    // numpy 的 rk_interval(max) 返回 [0, max] 区间的值
    // 因此 max_val = high - low - 1
    unsigned long max_val = static_cast<unsigned long>(high - low) - 1;
    if (high - low <= 1) return low;  // 空范围或单值

    // 计算位掩码: 大于等于 max_val 的最小 2^k - 1
    unsigned long mask = max_val;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;

    // 拒绝采样 (匹配 numpy rk_interval)
    unsigned long value;
    do {
        value = static_cast<unsigned long>(genrand_uint32()) & mask;
    } while (value > max_val);

    return low + static_cast<int>(value);
}

// ============================================================================
// Fisher-Yates 洗牌
//
// 匹配 numpy.random.shuffle 的行为:
//   for i from n-1 down to 1:
//       j = uniform_int(0, i+1)   // j ∈ [0, i]
//       swap(arr[i], arr[j])
// ============================================================================
void NumpyMT19937::shuffle(int* arr, int n) {
    if (n <= 1) return;
    for (int i = n - 1; i > 0; i--) {
        int j = uniform_int(0, i + 1);  // j ∈ [0, i]
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

// ============================================================================
// [0, 1) 浮点数: genrand_uint32() / 2^32
// ============================================================================
float NumpyMT19937::uniform_float() {
    return static_cast<float>(genrand_uint32()) / 4294967296.0f;
}
