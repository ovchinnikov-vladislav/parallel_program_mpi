#include <mpi/mpi.h>
#include <stdio.h>
#include "hello_mpi.h"

int start_hello_mpi(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    printf("Hello, MPI\n");

    MPI_Finalize();

    return 0;
}
