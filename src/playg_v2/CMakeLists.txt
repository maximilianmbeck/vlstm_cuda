cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES "80")

project(playg_v2 LANGUAGES CXX CUDA)

set(Torch_DIR /scratch/maximilian.beck/.conda/envs/xlstmdev1/lib/python3.11/site-packages/torch/share/cmake/Torch/)
# set(Torch_DIR /local00/bioinf/beck/envs/xlstmdev1/lib/python3.11/site-packages/torch/share/cmake/Torch/)

find_package(Torch REQUIRED)
# in order to avoid Python.h not found during make:
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})


add_library(funcs STATIC funcs.cc funcs.cu)
target_link_libraries(funcs ${TORCH_LIBRARIES})
target_link_libraries(funcs ${PYTHON_LIBRARIES})
set_property(TARGET funcs PROPERTY CXX_STANDARD 17)
set_target_properties(funcs PROPERTIES CUDA_ARCHITECTURES "80")
# set_property(TARGET funcs PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
# set_property(TARGET funcs PROPERTY CUDA_NVCC ${CMAKE_CUDA_ARCHITECTURES})

# # We need to explicitly state that we need all CUDA files in the particle
# # library to be built with -dc as the member functions could be called by
# # other libraries and executables
# set_target_properties( funcs
#                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
#                       )

add_executable(playg_v2 main.cpp)
target_link_libraries(playg_v2 "${TORCH_LIBRARIES}" funcs)
# target_link_libraries(playg_v2 funcs)
set_property(TARGET playg_v2 PROPERTY CXX_STANDARD 17)