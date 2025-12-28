#include <cmath>
#include <iostream>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

double heinitnormal(int nint) {
    static std::mt19937 gen{std::random_device{}()};
    std::normal_distribution<double> d(0.0, std::sqrt(2.0 / nint));
    return d(gen);
}
double heinituniform(int nint) {
    static std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<double> d(- std::sqrt(6.0 / nint), std::sqrt(6.0 / nint));
    return d(gen);
}
double lecunnnormal(int nint) {
    static std::mt19937 gen{std::random_device{}()};
    std::normal_distribution<double> d(0.0, 1/ nint);
    return d(gen);
}
double lecunnuniform(int nint) {
    static std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<double> d(- std::sqrt(3.0 / nint), std::sqrt(3.0 / nint));
    return d(gen);
}

double xaviernormal(int nin,int nout){
    static std::mt19937 gen{std::random_device{}()};
    std::normal_distribution<double> d(0.0, 2.0 / (nin+nout));
    return d(gen);
}

double xavieruniform(int nin,int nout){
    static std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<double> d(- std::sqrt(6 / (nin+nout)),std::sqrt(6 / (nin+nout)));
    return d(gen);
}

PYBIND11_MODULE(initialization, i) {
    i.doc() = "pybind11 example plugin";
    i.def("he_initalization_normal", &heinitnormal, "he initilzation in normal distrbution");
    i.def("he_initalization_uniform", &heinituniform, "he initilzation in uniform distrbution");
    i.def("xavier_initalization_normal", &xaviernormal, "xavier initilzation in normal distrbution");
    i.def("xavier_initalization_uniform", &xavieruniform, "xavier initilzation in uniform distrbution");
    i.def("le_cunn_initalization_normal", &lecunnnormal, "le cunn initilzation in normal distrbution");
    i.def("le_cunn_initalization_uniform", &lecunnuniform, "le cunn initilzation in uniform distrbution");
}
