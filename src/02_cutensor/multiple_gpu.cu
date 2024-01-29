#include <cstdlib>
#include <vector>

#include <common.h>
#include <cutensor.h>
#include <cutensorMg.h>
#include <cutensor_helpers.h>

int main() {
    cutensorMgHandle_t handle;
    std::vector<int32_t> devices{0, 1, 2, 3, 4, 5, 6, 7};
    cutensorMgCreate(&handle, 8, devices.data());
    cutensorMgDestroy(handle);
}