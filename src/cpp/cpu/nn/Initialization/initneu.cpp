#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "initneu.hpp"

namespace py = pybind11;

PYBIND11_MODULE(initialization, m) {
    m.doc() = "Weight initialization functions for minipytorch";

    m.def("he_normal", &he_normal, "He/Kaiming normal initialization (fan_in)");
    m.def("he_uniform", &he_uniform, "He/Kaiming uniform initialization (fan_in)");
    m.def("xavier_normal", &xavier_normal, "Xavier/Glorot normal (fan_in, fan_out)");
    m.def("xavier_uniform", &xavier_uniform, "Xavier/Glorot uniform (fan_in, fan_out)");
    m.def("lecun_normal", &lecun_normal, "LeCun normal initialization (fan_in)");
    m.def("lecun_uniform", &lecun_uniform, "LeCun uniform initialization (fan_in)");
}
