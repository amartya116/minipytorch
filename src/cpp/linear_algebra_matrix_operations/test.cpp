#include <iostream>
#include <string_view>  // C++17 feature

int main() {
    std::cout << "__cplusplus: " << __cplusplus << "\n";
#ifdef _MSVC_LANG
    std::cout << "_MSVC_LANG: " << _MSVC_LANG << "\n";
#endif

    // Test C++17 feature
    std::string_view sv = "Hello C++17!";
    std::cout << "C++17 string_view works: " << sv << "\n";

    std::cout << "If we see 202002 above, C++20 is working!\n";

    return 0;
}