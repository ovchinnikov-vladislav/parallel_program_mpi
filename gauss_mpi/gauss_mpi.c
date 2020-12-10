#include <stdio.h>
#include <stdlib.h>
#include <mpi/mpi.h>
#include "gauss_mpi.h"

void printM( float **a, int n ){
    int i, j;

    for( i=0; i< n; i++ ){
        for( j=0; j<=n; j++ )
            printf( " %14f", a[i][j] );
        printf("\n");
    }
}

int rank, size;

void forwardSubstitution(float **a, int n) {
    int i, j, k, max;
    float t;
    float *pt;


    for (i = 0; i < n; i++) {


        max = i;

        for (j = i + 1; j < n; j++)
            if ( abs(a[j][i]) > abs(a[max][i]) )
                max = j;

        pt = a[max];
        a[max] = a[i];
        a[i] = pt;


        for( k = i+1; k<n ; k++ ){
            t = a[k][i]/a[i][i];
            for( j = i; j<=n; j++ )
                a[k][j] -= a[i][j]*t;
        }

    }
}

void reverseElimination(float **a, int n) {
    int i, j;
    for (i = n - 1; i >= 0; i--) {
        a[i][n] = a[i][n] / a[i][i];
        a[i][i] = 1.0;
        for (j = i - 1; j >= 0; j--) {

            a[j][n] -= a[j][i] * a[i][n];
            a[j][i] = 0;
        }
    }
}

void testM( float **a, float **b, int n ){
    int i, j;
    float sum;
    printf("\n");

    for( i=0; i< n; i++ ){
        sum = 0.0;
        for( j=0; j<n; j++ )
            sum += b[i][j]*a[j][n];
        printf( " %14f=%14f [%14f]\n", b[i][n], sum, b[i][n]-sum  );
    }

}

void gauss(float **a, int n) {

    forwardSubstitution(a,n);
    reverseElimination(a,n);
}

int start_gauss_mpi(int argc, char *argv[]) {
    int i, j, k;

    int n=9;
    float **a, **b;
    double time;

    if( argc>1 ){
        sscanf( argv[1], "%d", &n );
    }


    MPI_Init (&argc, &argv);      /* starts MPI */
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);        /* get current process id */
    MPI_Comm_size (MPI_COMM_WORLD, &size);        /* get number of processes */
    printf( "Gauss for system[%d]\n", n );

    if( !rank ){
        float **ra;


        a = (float**) malloc(n*sizeof(float*));
        b = (float**) malloc(n*sizeof(float*));


        for (i = 0; i < n; i++) {
            a[i] = (float*) malloc((n+1)*sizeof(float));
            b[i] = (float*) malloc((n+1)*sizeof(float));
            for (j = 0; j < n+1; j++){
                a[i][j] = rand()%1000;
                b[i][j] = a[i][j];
            }
        }

        printM( a, n );

        time = MPI_Wtime();
        gauss(a, n);
        time = MPI_Wtime()-time;


        printf("Time[%d]: %lf\n", rank,time) ;
        printM( a, n );
        testM( a, b, n );
    }


    MPI_Finalize();
    return 0;
}