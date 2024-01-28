#ifndef CUDA_EXPERIMENTS_COMMON_H
#define CUDA_EXPERIMENTS_COMMON_H

#include <complex>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include <cutensor.h>

const uint32_t kAlignment =
    128; // Alignment of the global-memory device pointers (bytes)

#define CHECK_CUTENSOR_ERROR(val) checkCuTensor((val), #val, __FILE__, __LINE__)
template <typename T>
void checkCuTensor(T err, const char *const func, const char *const file,
                   const int line) {
    if (err != CUTENSOR_STATUS_SUCCESS) {
        std::cerr << "CUTENSOR Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cutensorGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <typename T> cutensorDataType_t get_cutensor_typ() {
    if (std::is_same<T, float>::value) {
        return CUTENSOR_R_32F;
    } else if (std::is_same<T, double>::value) {
        return CUTENSOR_R_64F;
    } else if (std::is_same<T, std::complex<float>>::value) {
        return CUTENSOR_C_32F;
    } else if (std::is_same<T, std::complex<double>>::value) {
        return CUTENSOR_C_64F;
    } else {
        throw std::runtime_error("Unsupported type");
    }
};

#endif // CUDA_EXPERIMENTS_COMMON_H
