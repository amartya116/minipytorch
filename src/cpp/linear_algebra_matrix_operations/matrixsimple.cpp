#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <cmath>
namespace py = pybind11;
using namespace std;

//internal function not to be included in Tensor.py
template <typename T>
bool is2d(const T &v) {
    if constexpr (std::is_same_v<typename T::value_type, typename T::value_type::value_type>) {
        return true;
    } else {
        return false;
    }
}
//internal function not to be included in Tensor.py

typedef std::vector<std::vector<double>> Matrix;

bool isSquare(const vector<vector<double>>& A){
    int rowsA=A.size();
    bool issqaure=false;
    for(int i=0;i<=rowsA;i++){
        if(A[0].size()==rowsA){
            issqaure=true;
        }
        else{
            return false;
        }
    }
    return true;
}
//internal function not to be included in Tensor.py
bool checkshape(const vector<vector<double>>& A, const vector<vector<double>>& B){
    int rowsA=A.size();
    int columnsA=A[0].size();
    int rowsB=B.size();
    int columnsB=B[0].size();
    if(rowsA==rowsB && columnsA==columnsB){
        return true;
    }
    else{
        return false;
    }
}
vector<vector<double>> add(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rows=A.size();
    int columns=A[0].size();
    vector<vector<double>> C(rows, vector<double>(columns));

    try{
        if(checkshape(A,B)){
            for(int i=0;i<rows;i++){
                for(int j=0;j<columns;j++){
                    C[i][j]=A[i][j]+B[i][j];
                }
            }
        }
        else{
            throw;
        }
    }
    catch(...){
        cout<<"Unable to add please check argument shapes if they are same";
    }
    return C;
}

vector<vector<double>> sub(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rows=A.size();
    int columns=A[0].size();

    vector<vector<double>> C(rows, vector<double>(columns));
    try{
        if(checkshape(A,B)){
            for(int i=0;i<rows;i++){
                for(int j=0;j<columns;j++){
                    C[i][j]=A[i][j]-B[i][j];
                }
            }
        }
        else{
            throw;
        }
    }
    catch(...){
        cout<<"Unable to subtract please check argument shapes if they are same";
    }
    return C;
}
vector<vector<double>> mul(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rows=A.size();
    int columns=A[0].size();

    vector<vector<double>> C(rows, vector<double>(columns));

   try{
        if(checkshape(A,B)){
            for(int i=0;i<rows;i++){ 
                for(int j=0;j<columns;j++){
                    C[i][j]=A[i][j]*B[i][j];
                }
            }
        }
        else{
            throw;
        }
    }
    catch(...){
        cout<<"Unable to multiply please check argument shapes if they are same";
    }
    return C;
}
vector<vector<double>> div_matrix(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rows=A.size();
    int columns=A[0].size();

    vector<vector<double>> C(rows, vector<double>(columns));

   try{
        if(checkshape(A,B)){
            for(int i=0;i<rows;i++){
                for(int j=0;j<columns;j++){
                    try{
                        if(B[i][j]==0){
                            C[i][j]=A[i][j]/B[i][j];
                        }
                        else{
                            throw;
                        }
                    }
                    catch(...){
                        cout<<"Denominator cannot be zero";
                    }
                }
            }
        }
        else{
            throw;
        }
    }
    catch(...){
        cout<<"Unable to divide please check argument shapes if they are same";
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
vector<vector<double>> sqrt_matrix(const vector<vector<double>>& A) {
    int rows=A.size();
    int columns=A[0].size();

    vector<vector<double>> C(rows, vector<double>(columns));

    for(int i=0;i<rows;i++){
        for(int j=0;j<columns;j++){
            try{
                if(A[i][j]>0){
                    C[i][j]=std::sqrt(A[i][j]);
                }
                else{
                    throw;
                }
            }
            catch(...){
                cout<<"Value cannot be imaginary";
            }
            
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
vector<vector<double>> log_matrix(const vector<vector<double>>& A) {
    int rows=A.size();
    int columns=A[0].size();

    vector<vector<double>> C(rows, vector<double>(columns));
    for(int i=0;i<rows;i++){
        for(int j=0;j<columns;j++){
            try{
                if(A[i][j]>0){
                    C[i][j]=log(A[i][j]);
                }
                else{
                    throw;
                }
        }
           catch(...){
            cout<<"value must be posítive or zero";
           }
        }
    }
    return C;
}
vector<double> sum1(vector<vector<double>> A,int axis){
    int rows=A.size();
    int columns=A[0].size();
    double sum=0;
    vector<double> result;
    if(axis==0){
    for(int i=0;i<rows;i++){
        for(int j=0;j<columns;j++){
            sum+=A[i][j];
        }
    }
        result.push_back(sum);
    }
    if(axis==1){
    for(int i=0;i<rows;i++){
        sum=0;
        for(int j=0;j<columns;j++){
            sum+=A[i][j];
        }
        result.push_back(sum);
    }
        
    }
    if(axis==2){
    for(int i=0;i<columns;i++){
        sum=0;
        for(int j=0;j<rows;j++){
            sum+=A[j][i];
        }
        result.push_back(sum);
    }
      
    }
     return result; 
}
vector<double> mul1(vector<vector<double>> A,int axis){
    int rows=A.size();
    int columns=A[0].size();
    double mul=1;
    vector<double> result;
    if(axis==0){
    for(int i=0;i<rows;i++){
        for(int j=0;j<columns;j++){
            mul*=A[i][j];
        }
    }
        result.push_back(mul);
    }
    if(axis==1){
    for(int i=0;i<rows;i++){
        mul=1;
        for(int j=0;j<columns;j++){
            mul*=A[i][j];
        }
        result.push_back(mul);
    }
        
    }
    if(axis==2){
    for(int i=0;i<columns;i++){
        mul=1;
        for(int j=0;j<rows;j++){
            mul*=A[j][i];
        }
        result.push_back(mul);
    }
      
    }
     return result; 
}
vector<double> mean(vector<vector<double>> A,int axis){
    int rows=A.size();
    int columns=A[0].size();
    double sum=0;
    vector<double> result;
    if(axis==0){
    for(int i=0;i<rows;i++){
        for(int j=0;j<columns;j++){
            sum+=A[i][j];
        }
    }
        result.push_back(sum/(rows*columns));
    }
    if(axis==1){
    for(int i=0;i<rows;i++){
        sum=0;
        for(int j=0;j<columns;j++){
            sum+=A[i][j];
        }
        result.push_back(sum/columns);
    }
        
    }
    if(axis==2){
    for(int i=0;i<columns;i++){
        sum=0;
        for(int j=0;j<rows;j++){
            sum+=A[j][i];
        }
        result.push_back(sum/rows);
    }
      
    }
     return result; 
}
vector<double> max(vector<vector<double>> A,int axis){
    int rows=A.size();
    int columns=A[0].size();
    vector<double> result;
    double max=0;
    if(axis==0){
    for(int i=0;i<rows;i++){
        for(int j=0;j<columns;j++){
            if(A[i][j]>max){
                max=A[i][j];
            }
           
        }
    }
        result.push_back(max);
    }
    if(axis==1){
    for(int i=0;i<rows;i++){
        max=0;
        for(int j=0;j<columns;j++){
              if(A[i][j]>max){
                max=A[i][j];
            }
        }
        result.push_back(max);
    }
        
    }
    if(axis==2){
    for(int i=0;i<columns;i++){
        max=0;
        for(int j=0;j<rows;j++){
            if(A[i][j]>max){
                max=A[i][j];
            }
        }
        result.push_back(max);
    }
      
    }
     return result; 
}
vector<double> min(vector<vector<double>> A,int axis){
    int rows=A.size();
    int columns=A[0].size();
    vector<double> result;
    double min=0;
    if(axis==0){
    for(int i=0;i<rows;i++){
        for(int j=0;j<columns;j++){
            if(A[i][j]<min){
                min=A[i][j];
            }
           
        }
    }
        result.push_back(min);
    }
    if(axis==1){
    for(int i=0;i<rows;i++){
        min=0;
        for(int j=0;j<columns;j++){
              if(A[i][j]<min){
                min=A[i][j];
            }
        }
        result.push_back(min);
    }
        
    }
    if(axis==2){
    for(int i=0;i<columns;i++){
        min=0;
        for(int j=0;j<rows;j++){
            if(A[i][j]<min){
                min=A[i][j];
            }
        }
        result.push_back(min);
    }
      
    }
     return result; 
}

vector<vector<int>> generateI(int size){
    vector<vector<int>> dynamicarray;
    for (int i = 0; i < size; i++) {
        {
            for(int j=0;j<size;j++){
                if(i==j){
                    dynamicarray[i][j]=1;
                }
                else{
                    dynamicarray[i][j]=0;
                }
                
            }
        }
    } 
     return dynamicarray;
}


PYBIND11_MAKE_OPAQUE(Matrix);

PYBIND11_MODULE(matrix_ops_simple, m) {
    m.doc() = "pybind11 example plugin";
    m.def("add", &add, "A function that adds two matrices");
    m.def("sub", &sub, "A function that subtracts two matrices");
    m.def("mul", &mul, "A function that multiplies two matrices element-wise");
    m.def("div", &div_matrix, "A function that divides two matrices element-wise");
    // m.def("sqr", py::cpp_function(&sqr), "A function that squares each element of a matrix");
    // m.def("sqrt", py::cpp_function(&sqrt_matrix), "A function that takes square root of each element of a matrix");
    m.def("exponent", &exponent, "A function that takes exponential of each element of a matrix");
    m.def("log", &log_matrix, "A function that takes logarithm of each element of a matrix");
    m.def("sum1", &sum1, "A function that sums elements along specified axis");
    m.def("mul1", &mul1, "A function that multiplies elements along specified axis");
    m.def("mean", &mean, "A function that computes mean along specified axis");
    m.def("max", &max, "A function that finds maximum along specified axis");
    m.def("min", &min, "A function that finds minimum along specified axis");
    m.def("generateI",&generateI,"A function to generate identity matrix");
}
