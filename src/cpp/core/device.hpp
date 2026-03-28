#include <iostream>
#include <map>
#include <cstdlib>
#include <vector>
namespace mylib::core
{
    enum device { CPU, GPU };
    struct CPUAllocator {
    std::map<size_t, std::vector<void*>> pool;
    void* allocate(size_t bytes){
        if(!pool[bytes].empty()){
            void* block = pool[bytes].back();   // get last element
            pool[bytes].pop_back();             // remove it
            return block; 
        }
        else{
            return malloc(bytes);
        }
    }
    void free(void* ptr,size_t bytes){
        pool[bytes].push_back(ptr);
    }
    size_t totalcached(){
        size_t totalsize=0;
        for(auto i:pool){
            totalsize+=i.second.size();
        }
        return totalsize;
    }
};
}