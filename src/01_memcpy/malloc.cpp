#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>

const size_t MB = 1024 * 1024;

int main() {
    float *buf;
    for (size_t i = (size_t)256 * MB; i <= (size_t)512 * 1024 * MB; i *= 2) {
        auto start = std::chrono::high_resolution_clock::now();
        buf = (float *)malloc(i * sizeof(float));
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        std::cout << "malloc " << i << " bytes took "
                  << (double)duration.count() / 1e6 << " ms" << std::endl;
        free(buf);
    }
}