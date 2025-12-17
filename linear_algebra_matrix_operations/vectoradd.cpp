#include <pybind11/pybind11.h>
#include <iostream>
#include <stdio.h>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

int sub(int i,int j){
    return i - j;
}

int* dyarray(int size){
    int* dynamicarray=new int[size];
    for (int i = 0; i < size; i++) {
        dynamicarray[i] = i * 10; // Example initialization
    } 
    std:: cout << "Dynamic Array Elements: ";
    for (int i = 0; i < size; i++) {
        std:: cout << dynamicarray[i] << " ";
}
    return dynamicarray;
}
PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin";
    m.def("add", &add, "A function that adds two numbers");
    m.def("sub", &sub, "A function that subs two numbers");
    m.def("darray",&dyarray,"A function to generate araay");
}
