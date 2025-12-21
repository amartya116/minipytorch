#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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
vector<vector<double>> log(const vector<vector<double>>& A) {
    int rows=A.size();
    int columns=A[0].size();

    vector<vector<double>> C(rows, vector<double>(columns));

    for(int i=0;i<rows;i++){
        for(int j=0;j<columns;j++){
            C[i][j]=log(A[i][j]);
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
   
PYBIND11_MODULE(matrix_ops_simple, m) {
    m.doc() = "pybind11 example plugin";
    m.def("add", &add, "A function that adds two numbers");
    m.def("sub", &sub, "A function that subs two numbers");
    m.def("generateI",&generateI,"A function to generate araay");
}
