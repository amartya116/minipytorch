#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdexcept>

namespace py = pybind11;
using namespace std;

// Static RNG (seeded once, thread-safe)
static std::mt19937& get_rng() {
    static std::mt19937 gen{42};
    return gen;
}

// He/Kaiming Initialization
double he_normal(int fan_in) {
    if (fan_in <= 0) throw std::invalid_argument("fan_in must be positive");
    static auto& gen = get_rng();
    std::normal_distribution<double> d(0.0, std::sqrt(2.0 / fan_in));
    return d(gen);
}

double he_uniform(int fan_in) {
    if (fan_in <= 0) throw std::invalid_argument("fan_in must be positive");
    static auto& gen = get_rng();
    double bound = std::sqrt(6.0 / fan_in);
    std::uniform_real_distribution<double> d(-bound, bound);
    return d(gen);
}

// Xavier/Glorot Initialization
double xavier_normal(int fan_in, int fan_out) {
    if (fan_in <= 0 || fan_out <= 0) throw std::invalid_argument("fan_in and fan_out must be positive");
    static auto& gen = get_rng();
    std::normal_distribution<double> d(0.0, std::sqrt(2.0 / (fan_in + fan_out)));
    return d(gen);
}

double xavier_uniform(int fan_in, int fan_out) {
    if (fan_in <= 0 || fan_out <= 0) throw std::invalid_argument("fan_in and fan_out must be positive");
    static auto& gen = get_rng();
    double bound = std::sqrt(6.0 / (fan_in + fan_out));
    std::uniform_real_distribution<double> d(-bound, bound);
    return d(gen);
}

// LeCun Initialization
double lecun_normal(int fan_in) {
    if (fan_in <= 0) throw std::invalid_argument("fan_in must be positive");
    static auto& gen = get_rng();
    std::normal_distribution<double> d(0.0, std::sqrt(1.0 / fan_in));
    return d(gen);
}

double lecun_uniform(int fan_in) {
    if (fan_in <= 0) throw std::invalid_argument("fan_in must be positive");
    static auto& gen = get_rng();
    double bound = std::sqrt(3.0 / fan_in);
    std::uniform_real_distribution<double> d(-bound, bound);
    return d(gen);
}

PYBIND11_MODULE(initialization, m) {
    m.doc() = "Weight initialization functions for minipytorch";

    m.def("he_normal", &he_normal, "He/Kaiming normal initialization (fan_in)");
    m.def("he_uniform", &he_uniform, "He/Kaiming uniform initialization (fan_in)");
    m.def("xavier_normal", &xavier_normal, "Xavier/Glorot normal (fan_in, fan_out)");
    m.def("xavier_uniform", &xavier_uniform, "Xavier/Glorot uniform (fan_in, fan_out)");
    m.def("lecun_normal", &lecun_normal, "LeCun normal initialization (fan_in)");
    m.def("lecun_uniform", &lecun_uniform, "LeCun uniform initialization (fan_in)");
}