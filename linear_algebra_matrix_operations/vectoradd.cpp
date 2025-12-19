#include <pybind11/pybind11.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <cmath>
namespace py = pybind11;
using namespace std;

vector<vector<double>> add(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rows=A.size();
    int columns=A[0].size();

    vector<vector<double>> C(rows, vector<double>(columns));

    for(int i=0;i<rows;i++){
        for(int j=0;j<columns;j++){
            C[i][j]=A[i][j]+B[i][j];
        }
    }
    return C;
}

vector<vector<double>> sub(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rows=A.size();
    int columns=A[0].size();

    vector<vector<double>> C(rows, vector<double>(columns));

    for(int i=0;i<rows;i++){
        for(int j=0;j<columns;j++){
            C[i][j]=A[i][j]-B[i][j];
        }
    }
    return C;
}
vector<vector<double>> mul(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rows=A.size();
    int columns=A[0].size();

    vector<vector<double>> C(rows, vector<double>(columns));

    for(int i=0;i<rows;i++){
        for(int j=0;j<columns;j++){
            C[i][j]=A[i][j]*B[i][j];
        }
    }
    return C;
}
vector<vector<double>> div(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rows=A.size();
    int columns=A[0].size();

    vector<vector<double>> C(rows, vector<double>(columns));

    for(int i=0;i<rows;i++){
        for(int j=0;j<columns;j++){
            C[i][j]=A[i][j]/B[i][j];
        }
    }
    return C;
}
vector<vector<double>> sqr(const vector<vector<double>>& A) {
    int rows=A.size();
    int columns=A[0].size();

    vector<vector<double>> C(rows, vector<double>(columns));

    for(int i=0;i<rows;i++){
        for(int j=0;j<columns;j++){
            C[i][j]=A[i][j]*A[i][j];
        }
    }
    return C;
}
vector<vector<double>> sqrt(const vector<vector<double>>& A) {
    int rows=A.size();
    int columns=A[0].size();

    vector<vector<double>> C(rows, vector<double>(columns));

    for(int i=0;i<rows;i++){
        for(int j=0;j<columns;j++){
            C[i][j]=sqrt(A[i][j]);
        }
    }
    return C;
}
vector<vector<double>> exponent(const vector<vector<double>>& A) {
    int rows=A.size();
    int columns=A[0].size();

    vector<vector<double>> C(rows, vector<double>(columns));

    for(int i=0;i<rows;i++){
        for(int j=0;j<columns;j++){
            C[i][j]=exp(A[i][j]);
        }
    }
    return C;
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
    m.def("dyarray",&dyarray,"A function to generate araay");
}
