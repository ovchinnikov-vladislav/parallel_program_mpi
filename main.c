//#include "hello_mpi/hello_mpi.h"
//#include "gauss/gauss.h"
//#include "gauss_mpi/gauss_mpi.h"
//#include "gauss_parall_mpi_1/gauss_parall_mpi_1.h"
#include "gauss_parall_mpi_2/gauss_parall_mpi_2.h"

int main(int argc, char** argv) {
    //start_hello_mpi(argc, argv);
    //start_gauss(argc, argv);
    //start_gauss_mpi(argc, argv);
    //start_gauss_parallel_1_mpi(argc, argv);
    start_gauss_parall_mpi_2(argc, argv);

    return 0;
}
