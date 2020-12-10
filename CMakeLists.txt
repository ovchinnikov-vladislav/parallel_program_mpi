cmake_minimum_required(VERSION 3.17)
project(parallel_programming)

set(CMAKE_C_STANDARD 99)

add_executable(parallel_programming main.c hello_mpi/hello_mpi.c gauss/gauss.c gauss_mpi/gauss_mpi.c
        gauss_parall_mpi_1/gauss_parall_mpi_1.c gauss_parall_mpi_2/gauss_parall_mpi_2.c)