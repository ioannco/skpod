cmake_minimum_required(VERSION 3.27)
project(skpod C)

set(CMAKE_C_STANDARD 11)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(omp_for var75_omp_for.c
        task.c)
add_executable(omp_task var75_omp_task.c)
add_executable(vanilla var75.c)
