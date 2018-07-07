cmake_minimum_required(VERSION 3.10)
project(Parallel_Inversion C)

set(CMAKE_C_STANDARD 99)
#set(CMAKE_C_COMPILER /usr/bin/mpicc)
set(SOURCE_FILES main.c)

add_executable(Parallel_Inversion ${SOURCE_FILES})
find_package(GSL REQUIRED)
target_link_libraries(Parallel_Inversion GSL::gsl GSL::gslcblas)
