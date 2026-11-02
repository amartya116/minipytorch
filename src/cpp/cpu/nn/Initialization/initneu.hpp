#pragma once
#include <cmath>
#include <random>
#include <stdexcept>

// Static RNG (seeded once, thread-safe)
inline std::mt19937& get_rng() {
    static std::mt19937 gen{42};
    return gen;
}

// He/Kaiming Initialization
inline double he_normal(int fan_in) {
    if (fan_in <= 0) throw std::invalid_argument("fan_in must be positive");
    static auto& gen = get_rng();
    std::normal_distribution<double> d(0.0, std::sqrt(2.0 / fan_in));
    return d(gen);
}

inline double he_uniform(int fan_in) {
    if (fan_in <= 0) throw std::invalid_argument("fan_in must be positive");
    static auto& gen = get_rng();
    double bound = std::sqrt(6.0 / fan_in);
    std::uniform_real_distribution<double> d(-bound, bound);
    return d(gen);
}

// Xavier/Glorot Initialization
inline double xavier_normal(int fan_in, int fan_out) {
    if (fan_in <= 0 || fan_out <= 0) throw std::invalid_argument("fan_in and fan_out must be positive");
    static auto& gen = get_rng();
    std::normal_distribution<double> d(0.0, std::sqrt(2.0 / (fan_in + fan_out)));
    return d(gen);
}

inline double xavier_uniform(int fan_in, int fan_out) {
    if (fan_in <= 0 || fan_out <= 0) throw std::invalid_argument("fan_in and fan_out must be positive");
    static auto& gen = get_rng();
    double bound = std::sqrt(6.0 / (fan_in + fan_out));
    std::uniform_real_distribution<double> d(-bound, bound);
    return d(gen);
}

// LeCun Initialization
inline double lecun_normal(int fan_in) {
    if (fan_in <= 0) throw std::invalid_argument("fan_in must be positive");
    static auto& gen = get_rng();
    std::normal_distribution<double> d(0.0, std::sqrt(1.0 / fan_in));
    return d(gen);
}

inline double lecun_uniform(int fan_in) {
    if (fan_in <= 0) throw std::invalid_argument("fan_in must be positive");
    static auto& gen = get_rng();
    double bound = std::sqrt(3.0 / fan_in);
    std::uniform_real_distribution<double> d(-bound, bound);
    return d(gen);
}
