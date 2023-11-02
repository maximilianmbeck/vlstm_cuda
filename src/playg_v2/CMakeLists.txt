cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(playg_v2 LANGUAGES CXX CUDA)

set(Torch_DIR /scratch/maximilian.beck/.conda/envs/xlstmdev1/lib/python3.11/site-packages/torch/share/cmake/Torch/)
# set(Torch_DIR /local00/bioinf/beck/envs/xlstmdev1/lib/python3.11/site-packages/torch/share/cmake/Torch/)

find_package(Torch REQUIRED)

add_executable(playg_v2 main.cpp)
target_link_libraries(playg_v2 "${TORCH_LIBRARIES}")
set_property(TARGET playg_v2 PROPERTY CXX_STANDARD 17)