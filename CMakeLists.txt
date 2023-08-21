cmake_minimum_required(VERSION 3.10)
project(CUDA_RAY_TRACING LANGUAGES CUDA CXX)
include(CheckLanguage)
check_language(CUDA)
SET(SRC_LIST camera.h hittable.h hittable_list.h kernel.cu material.h ray.h sphere.h utility.h vec3.h)
set(CMAKE_CXX_STANDARD 20)
SET(CUDA_NVCC_FLAGS -std=c++17)
ADD_EXECUTABLE(cuda_ray_tracing ${SRC_LIST})

