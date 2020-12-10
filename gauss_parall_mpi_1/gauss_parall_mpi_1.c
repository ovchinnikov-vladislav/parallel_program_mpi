#include <stdio.h>
#include <stdlib.h>
#include <mpi/mpi.h>
#include <string.h>
#include "gauss_parall_mpi_1.h"

int rank, size, dbgline;
double  *x_global;
#define DBGLN(xline)  printf("%02d:%04d:%s\n",rank,++dbgline,xline);fflush( stdout )

void printC( double *a, int n, int in, char *key ){
    int i, j;
    char *pbuf, rbuf[20];
    pbuf = (char*) malloc( 40 + (n+1)*16 );

    for( i=0; i< in; i++ ){
        pbuf[0] = 0;

        for( j=0; j<=n; j++ ){
            sprintf( rbuf, " %14lf", a[i*(n+1)+j] );
            strcat( pbuf, rbuf );
        }
        DBGLN( pbuf );
    }

    free(pbuf);
}

void printM( double **a, int n ){
    int i, j;
    char *pbuf, rbuf[20];
    pbuf = (char*) malloc( 40 + (n+1)*16 );

    for( i=0; i<n; i++ ){
        pbuf[0] = 0;

        for( j=0; j<=n; j++ ){
            sprintf( rbuf," %14f", a[i][j] );
            strcat( pbuf, rbuf );
        }
        DBGLN( pbuf );
    }

    free(pbuf);
}

void printKey( double *a, int n ){
    int j;
    char *pbuf, rbuf[20];
    pbuf = (char*) malloc( 40 + (n+1)*16 );
    strcpy( pbuf, "Key=" );


    for( j=0; j<n; j++ ){
        sprintf( rbuf," %14f", a[j] );
        strcat( pbuf, rbuf );
    }
    DBGLN( pbuf );

    free(pbuf);
}

void forwardSubstitution(double *a, int n, int local_n) {
    int i_local, i_global, j, k, drank,i;
    double *keystring, *buffer;
    double t;
    char rr[40], *cbuf;
    cbuf = ( char* ) malloc( 100 );

    sprintf( cbuf, "forwardSubstitution n=%d local_n=%d a[%8lx %8lx]",
             n, local_n,
             (unsigned long int) a,
             (unsigned long int) &a[local_n*n-1] );
    DBGLN( cbuf );

    i_local = 0;
    buffer = (double*) malloc( sizeof(double)*(n+1) );

    for (i_global = 0; i_global < n; i_global++) {
        drank = i_global % size;
        if( n<=100 ){
            sprintf( cbuf, "I[%d] in Main Cicle start %d[%d] ", rank, i_global, i_local );
            DBGLN( cbuf );
        }

        if(  drank == rank ){
            keystring = & a[i_local*(n+1)];

            if( n<=100 ){
                sprintf( cbuf, "I[%d]=[%d] am main in %d key=[%14f %14f .. %14f]", rank, drank, i_global, keystring[0], keystring[1], keystring[9] );
                DBGLN( cbuf );
                printKey( keystring, n+1 );
            }

            MPI_Bcast( keystring, n+1, MPI_DOUBLE, drank, MPI_COMM_WORLD );

            i = ++i_local;
        }
        else {

            if( n<=10 ){
                sprintf( cbuf, "I[%d] Waiting from[%d] string %d", rank, drank, i_global );
                DBGLN( cbuf );
            }

            MPI_Bcast( buffer, n+1, MPI_DOUBLE, drank, MPI_COMM_WORLD );
            keystring = buffer;

            if( n<=10 ){
                sprintf( cbuf, "I[%d] am ready in %d key=[%14f %14f ..]", rank, i_global, keystring[0], keystring[1] );
                DBGLN( cbuf );
                printKey( keystring, n+1 );
            }

            i = i_local;
        }


        if( n<=10 ){
            sprintf( cbuf, "I[%d] do Main Cicle %d[%d]  key[%8lx]=[%14f %14f .. %14f]", rank,
                     i_global, i_local,
                     (unsigned long int)  keystring,
                     keystring[0], keystring[1], keystring[n] );

            DBGLN( cbuf );
            sprintf( cbuf, "Key values %14f[%8lx] x = %14f[%8lx]",
                     keystring[i_global], (unsigned long int) &keystring[i_global],
                     keystring[n], (unsigned long int) &keystring[n] );
            DBGLN( cbuf );
        }

//		sprintf( rr, "It%04dS", i_global );
//        	printC( a, n, local_n, rr );

        if( keystring[i_global]==0.0 ){
            printf("I[%d] Found Zero %d", rank, i_global);
            exit;
        }
        for( k = i; k< local_n ; k++ ){

            t =  a[k*(n+1)+i_global] / keystring[i_global];

            if( n<=10 ){
                sprintf( cbuf, " Div = %f", t );
                DBGLN( cbuf );
            }

            for( j = i_global; j<=n; j++ ){
                a[k*(n+1)+j] -= t*keystring[j];

                if( n<=10 ){
                    sprintf( cbuf, " New[%d][%d] = %f", k, j, a[k*(n+1)+j] );
                    DBGLN( cbuf );
                }

            }
        }

        if( n<=10 ){
            MPI_Barrier( MPI_COMM_WORLD );
            sprintf( cbuf, "Main Cicle  end  %d[%d]", i_global, rank );
            DBGLN( cbuf );
//		   sprintf( rr, "It%04dE", i_global );
//        	   printC( a, n, local_n, rr );
        }

    }
    free(buffer);
    free(cbuf);
}

/*
void forwardSubstitution(double **a, int n) {
        int i, j, k, max;
        double t;
	double *pt;
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
*/

void reverseElimination( double *a, int n, int local_n ) {
    int i_local, i_global, j, k, drank,i;
    char rr[40], *cbuf;
    double x;

    cbuf = ( char* ) malloc( 200 );
    sprintf( cbuf, "reverseElimination n=%d local_n=%d a[%8lx %8lx]",
             n, local_n,
             (unsigned long int) a,  (unsigned long int) a+(local_n*n-1 ) );
    DBGLN( cbuf );

    i_local = local_n-1;
    for (i_global = n - 1; i_global > 0; i_global--) {

        drank = i_global % size;
        if( n<=1000 ){
            sprintf( cbuf, "I[%d] start [%d]", rank, i_global );
            DBGLN( cbuf );
        }

        if(  drank == rank ){

            if( n<=1000 ){
                sprintf( cbuf, "x[%d]in  KEY=%8lx %d[%d]=  %14f[%8lx]  / %14f[%8lx] = %14f",
                         i_global,
                         (unsigned long int) &a[i_local*(n+1)], rank, i_local,
                         a[i_local*(n+1)+n], (unsigned long int)  &a[i_local*(n+1)+n],
                         a[i_local*(n+1)+i_global], (unsigned long int) &a[i_local*(n+1)+i_global],
                         a[i_local*(n+1)+n] / a[i_local*(n+1)+i_global]  );
                DBGLN( cbuf );
            }

//                   a[i_local][n] = a[i_local][n] / a[i_local][i_global];
            if( a[i_local*(n+1)+i_global] == 0 ){
                sprintf( cbuf, "ERROR  %14f[%8lx]  / %14f[%8lx]",
                         a[i_local*(n+1)+n], (unsigned long int) &a[i_local*(n+1)+n],
                         a[i_local*(n+1)+i_global], (unsigned long int)&a[i_local*(n+1)+i_global] );
                DBGLN( cbuf );
                exit;
            }
            a[i_local*(n+1)+n] = a[i_local*(n+1)+n] / a[i_local*(n+1)+i_global];

            a[i_local*(n+1)+i_global] = 1.0;
            x = a[i_local*(n+1)+n];
//                   a[i_local][i_global] = 1.0;
//                   x = a[i_local][n];
            i_local--;

            if( n<=1000 ){
                sprintf( cbuf, "I[%d]=[%d] am main x%d=%14f", rank,
                         drank, i_global, x );
                DBGLN( cbuf );
            }
//		   x = 1.0;
            MPI_Bcast( &x, 1, MPI_DOUBLE, drank, MPI_COMM_WORLD );

            if( n<=1000 ){
                sprintf( cbuf, "I[%d]=[%d] sand x%d=%14f", rank,
                         drank, i_global, x );
                DBGLN( cbuf );
            }
        }
        else {
//		   x = 1.0;
            if( n<=1000 ){
                sprintf( cbuf, "I[%d] waiting from [%d] x%d", rank,
                         drank, i_global );
                DBGLN( cbuf );
            }

            MPI_Bcast( &x, 1, MPI_DOUBLE, drank, MPI_COMM_WORLD );

            if( n<=1000 ){
                sprintf( cbuf, "I[%d] from [%d] got x%d=%14f", rank,
                         drank, i_global, x );
                DBGLN( cbuf );
            }
        }


        if( n<=1000 ){
            sprintf( cbuf, "I[%d] do job with x%d=%14f", rank,
                     i_global, x );
            DBGLN( cbuf );
        }
        if( !rank )
            x_global[i_global] = x;

        for (j = i_local; j >= 0; j--) {
//                     a[j][n] -= a[j][i] * a[i][n];
            if( n<=100 ){

                sprintf( cbuf, "Work[%d]  a[%d n]-a[%d,%d]*x%d=   -  %14f - %14f * %14f", rank, j, j, i_global, i_global,
                         a[j*(n+1)+n], a[j*(n+1)+i_global], x );
                DBGLN( cbuf );
            }

            a[j*(n+1)+n] -= a[j*(n+1)+i_global] * x;


//                     a[j][i] = 0;
            a[j*(n+1)+i_global] = 0.0;
        }
        if( n<=1000 ){
            sprintf( cbuf, "I[%d] did [%d]", rank, i_global );
            DBGLN( cbuf );
        }
        if( i_global > 1 ){
            MPI_Barrier( MPI_COMM_WORLD );
        }

    }
    if( !rank ){
        if( a[0] == 0 ){
            sprintf( cbuf, "ERROR  %14f[%8lx]  / %14f[%8lx]",
                     a[n], (unsigned long int)&a[n],
                     a[0], (unsigned long int)&a[0] );
            DBGLN( cbuf );
            exit;
        }
        a[n] = a[n] / a[0];
        a[0] = 1.0;
        x = a[n];
        x_global[0] = x;



    }

    free(cbuf);
}

/*
void reverseElimination(double **a, int n) {
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
*/
void testM( double *x, double **b, int n ){
    int i, j;
    double sum;
    char cbuf[80];

    DBGLN( "Test Solution" );

    for( i=0; i< n; i++ ){
        sum = 0.0;
        for( j=0; j<n; j++ )
            sum += b[i][j]*x[j];

        sprintf( cbuf, "test B%d  %14f=%14f [%14f]", i, b[i][n], sum, b[i][n]-sum );
        DBGLN( cbuf );
    }

}

void par_gauss(double *a, int n, int local_n ) {
    int i, j;
    char cbuf[80];

    sprintf( cbuf, "par_gauss in %d [%10f %10f ..]", rank, a[0], a[1] );
    DBGLN( cbuf );

    forwardSubstitution( a, n, local_n );
    reverseElimination( a, n, local_n );
}

int start_gauss_parallel_1_mpi(int argc, char *argv[]) {
    int i, ii, j, ip, pbase, psize, psizeadd;
    char cbuf[80];

    int n=100;
    double **b, *buf;
    double time;
    MPI_Status st;


    MPI_Init (&argc, &argv);      /* starts MPI */
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);        /* get current process id */
    MPI_Comm_size (MPI_COMM_WORLD, &size);        /* get number of processes */
    if( argc>1 ){
        sscanf( argv[1], "%d", &n );
    }

    sprintf( cbuf, "Start Gausss Paralllel[%d] from process %d of %d", n, rank, size );
    DBGLN( cbuf );


    if( !rank ){
        int psize1;
        sprintf( cbuf, "0 Generating Matrix rank=%d", rank );
        DBGLN( cbuf );

        b = (double**) malloc(n*sizeof(double*));
        if( !b ){
            sprintf( cbuf, "B Zero !!  Error" );
            DBGLN( cbuf );
            return (2);
        }

        for (i = 0; i < n; i++) {
            b[i] = (double*) malloc( (n+1)*sizeof(double) );

            if( !(b[i]) ){
                sprintf( cbuf, "B[%d] Zero !!  Error", i );
                DBGLN( cbuf );
                return (3);
            }

            for (j = 0; j < n+1; j++){
                b[i][j] = rand() % 1000;
                if( n< 400)
                    printf(" %14f", b[i][j] );
            }
            if( n< 400)
                printf("\n");
        }
        x_global = (double*) malloc( (n+10)*sizeof(double) );
        if( !x_global ){
            sprintf( cbuf, "X Zero !!  Error" );
            DBGLN( cbuf );
            return (4);
        }

        psize = ( n / size );
        psizeadd = n % size;

        sprintf( cbuf, "Matrix B ready n=%d (psize=%d ++ %d ) size=%d", n, psize, psizeadd, size );
        DBGLN( cbuf );
        if( n< 400)
            printM( b, n );

        pbase = psize + ((psizeadd)?1:0);
        sprintf( cbuf, "pbase=%d   psize=%d  psizeadd= %d", pbase, psize, psizeadd );
        DBGLN( cbuf );

        buf = (double* ) malloc( pbase*(n+1)*sizeof(double) );
        if( !buf ){
            printf("Buf is NULL %d\n", rank  );
            return (20);
        }


        for( ip=1; ip< size; ip++ ){

            psize1 = psize + ((psizeadd>ip)?1:0);
            sprintf( cbuf, "Make chank for proc %d of %d  pbase=%d -> [%d]", ip, size, pbase, psize1 );
            DBGLN( cbuf );

            ii=0;
            for( i=pbase; i<(pbase+psize1); i++ ){
                for( j=0; j<=n; j++ )
                    buf[ ii*(n+1)+j] = b[i][j];
                ii++;
            }

            sprintf( cbuf, "send to %d  strings %d[%d]  [%10f %10f ..]", ip, pbase, psize, buf[0], buf[1] );
            DBGLN( cbuf );

            MPI_Send( buf, psize1*(n+1), MPI_DOUBLE, ip, 10, MPI_COMM_WORLD );

            pbase += psize1;
        }

        psize += (psizeadd)?1:0;
        for( i=0; i<psize; i++ ){
            for( j=0; j<=n; j++ )
                buf[ i*(n+1)+j] = b[i][j];
        }

        sprintf( cbuf, "0 redy" );
        DBGLN( cbuf );

    } else {
        psize = ( n / size );
        psizeadd = n % size;
        psize += (psizeadd>rank)?1:0;

        buf = (double* ) malloc( psize*sizeof(double)*(n+1) );
        if( !buf ){
            printf("Buf is NULL %d\n", rank  );
            return (10);
        }

        sprintf( cbuf, "recving in %d  strings [%d]", rank, psize );
        DBGLN( cbuf );

        MPI_Recv( buf, psize*(n+1), MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, &st );

        sprintf( cbuf, "recv in %d  strings [%d]", rank, psize );
        DBGLN( cbuf );
    }

//        MPI_Barrier( MPI_COMM_WORLD );

    time = MPI_Wtime();
    par_gauss(buf, n, psize );
    time = MPI_Wtime()-time;


    sprintf( cbuf, "Time[%d]: %f", rank,time) ;
    DBGLN( cbuf );

    MPI_Barrier( MPI_COMM_WORLD );

    if( rank==0 ){
        testM( x_global, b, n );
        free( x_global );
    }

    MPI_Barrier( MPI_COMM_WORLD );

    sprintf( cbuf, "Exit MPI[%d]", rank) ;
    DBGLN( cbuf );

    MPI_Finalize();

    sprintf( cbuf, "Success[%d]", rank) ;
    DBGLN( cbuf );

    return 0;
}

