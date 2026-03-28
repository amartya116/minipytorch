#include <stdio.h>
#include <vector>
#include <string>
using namespace std;
namespace mylib::core {
enum Dtypes {
	uint8=0,
	int8=1,
	int32=2,
	int64=3,
	float16=4,
	float32=5,
	float64=6
};
int itemsize(Dtypes Dtype) {
    int sizeneeded;
	switch (Dtype) {
	case float32:
		sizeneeded=4;
		break;
	case float64:
		sizeneeded=8;
		break;
	case float16:
		sizeneeded=2;
		break;
	case uint8:
		sizeneeded=1;
		break;
	case int8:
		sizeneeded=1;
		break;
	case int32:
		sizeneeded=4;
		break;
	case int64:
		sizeneeded=8;
		break;
	}
	return sizeneeded;

}
string name(Dtypes Dtype){
    switch (Dtype) {
	case float32:
		return "float32";
	case float64:
		return "float64";
	case float16:
		return "float16";
	case uint8:
		return "uint8";
	case int8:
		return "int8";
	case int32:
		return "int32";
	case int64:
		return "int64";
	}
 }

Dtypes promote(Dtypes Dtype1,Dtypes Dtype2){
    if(Dtype1>Dtype2){
        return Dtype1;
    }
    else{
        return Dtype2;
    }
 }

}