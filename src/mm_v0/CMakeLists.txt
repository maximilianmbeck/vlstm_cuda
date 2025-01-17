cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# set(CMAKE_CUDA_ARCHITECTURES "80")
project(vlstm_v1 LANGUAGES CXX CUDA)

# set(Torch_DIR /home/max/miniconda3/envs/xlstm/lib/python3.11/site-packages/torch/share/cmake/Torch/)
set(Torch_DIR /home/max/cpplibs/libtorch/share/cmake/Torch/)

# set(Torch_DIR /local00/bioinf/beck/envs/xlstmdev1/lib/python3.11/site-packages/torch/share/cmake/Torch/)
find_package(Torch REQUIRED)

# in order to avoid Python.h not found during make:
# find_package(pybind11 REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# message(WARNING "Torch_DIR: ") # ${TORCH_LIBRARIES})
# message(WARNING "Python dirs: ") # ${PYTHON_INCLUDE_DIRS})
add_library(kernels STATIC interface.cc kernels.cu) # ! original version

# pybind11_add_module(kernels SHARED interface.cc kernels.cu) # ! try pybind11
target_link_libraries(kernels ${PYTHON_LIBRARIES})
target_link_libraries(kernels ${TORCH_LIBRARIES})

# target_link_libraries(kernels ${TORCH_PYTHON_LIBRARIES})
set_property(TARGET kernels PROPERTY CXX_STANDARD 17)

# set_target_properties(kernels PROPERTIES CUDA_ARCHITECTURES "80")

# set_property(TARGET kernels PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

# # We need to explicitly state that we need all CUDA files in the particle
# # library to be built with -dc as the member functions could be called by
# # other libraries and executables
set_target_properties(kernels
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" kernels)

# target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARIES}" kernels)
# target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
# target_link_libraries(${PROJECT_NAME} ${TORCH_PYTHON_LIBRARIES})
# target_link_libraries(${PROJECT_NAME} kernels)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)