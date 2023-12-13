cmake_minimum_required(VERSION 3.27)
project(skpod C)

set(CMAKE_C_STANDARD 11)

add_executable(skpod var75_omp_for.c)
