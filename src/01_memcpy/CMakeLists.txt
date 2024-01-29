add_executable(cpu cpu.cpp)
add_executable(single_gpu single_gpu.cu)
add_executable(multiple_gpu multiple_gpu.cu)

target_link_libraries(single_gpu nvToolsExt)
target_link_libraries(multiple_gpu nvToolsExt mpi mpi_cxx)
