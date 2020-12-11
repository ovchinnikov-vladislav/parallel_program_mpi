#include <mpi/mpi.h>
#include <stdlib.h>
#include <stdio.h>


int get_chunk(int total, int commsize, int rank) {
    int n = total;
    int q = n / commsize;
    if (n % commsize) {
        q++;
    }
    int r = commsize * q - n;

    int chunk = q;
    if (rank >= commsize - r) {
        chunk = q - 1;
    }
    return chunk;
}

void print(int rank, int commsize, int *rows, int nrows) {
    // Вывод в порядке proc 0, 1, 2, ... , p - 1
    MPI_Recv(NULL, 0, MPI_INT, (rank > 0) ? rank - 1 : MPI_PROC_NULL, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Proc %d: ", rank);
    for (int i = 0; i < nrows; i++) {
        printf("%d ", rows[i]);
    }
    printf("\n");
    MPI_Ssend(NULL, 0, MPI_INT, rank != commsize - 1 ? rank + 1 : MPI_PROC_NULL, 0, MPI_COMM_WORLD);
}

int start_gauss_parall_mpi_2(int argc, char **argv) {
    int n = 10;

    if (argc == 2) {
    	n = atoi(argv[1]);
    }

    int rank, commsize;
    MPI_Init(&argc, &argv);
    double t = MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);

    int nrows = get_chunk(n, commsize, rank);
    int *rows = malloc(sizeof(*rows) * nrows);          // Номера локальных строк

    // Матрица дополнена столбцом для вектора b
    double *a = malloc(sizeof(*a) * nrows * (n + 1));
    double *x = malloc(sizeof(*x) * n);
    double *tmp = malloc(sizeof(*tmp) * (n + 1));

    // Инициализация системы уравнений случайными числами
    for (int i = 0; i < nrows; i++) {
        rows[i] = rank + commsize * i;
        srand(rows[i] * (n + 1));
        for (int j = 0; j < n; j++) {
            a[i * (n + 1) + j] = rand() % 100 + 1;
        }
        a[i * (n + 1) + n] = rand() % 100 + 1;
    }

    print(rank, commsize, rows, nrows);

    // Прямой ход решения
    int row = 0;
    for (int i = 0; i < n - 1; i++) {
        // Исключаем x_i
        if (i == rows[row]) {
            // Рассылаем строку i, находящуюся в памяти текущего процесса
            MPI_Bcast(&a[row * (n + 1)], n + 1, MPI_DOUBLE, rank, MPI_COMM_WORLD);
            for (int j = 0; j <= n; j++) {
                tmp[j] = a[row * (n + 1) + j];
            }
            row++;
        } else {
            MPI_Bcast(tmp, n + 1, MPI_DOUBLE, i % commsize, MPI_COMM_WORLD);
        }

        // Вычитаем принятую строку из уравнений, хранящихся в текущем процессе
        for (int j = row; j < nrows; j++) {
            double scalling = a[j * (n + 1) + i] / tmp[i];
            for (int k = i; k < n + 1; k++) {
                a[j * (n + 1) + k] -= scalling * tmp[k];
            }
        }
    }

    // Инициализация неизвестных
    row = 0;
    for (int i = 0; i < n; i++) {
        x[i] = 0;
        if (i == rows[row]) {
            x[i] = 0;
            if (i == rows[row]) {
                x[i] = a[row * (n + 1) + n];
                row++;
            }
        }
    }

    // Обратный ход решения
    row = nrows - 1;
    for (int i = n - 1; i > 0; i--) {
        if (row >= 0) {
            if (i == rows[row]) {
                // Передаем найдненное x_i
                x[i] /= a[row * (n + 1) + i];
                MPI_Bcast(&x[i], 1, MPI_DOUBLE, rank, MPI_COMM_WORLD);
                row--;
            } else {
                MPI_Bcast(&x[i], 1, MPI_DOUBLE, i % commsize, MPI_COMM_WORLD);
            }
        } else {
            MPI_Bcast(&x[i], 1, MPI_DOUBLE, i % commsize, MPI_COMM_WORLD);
        }

        // Корректировка локальных x_i
        for (int j = 0; j <= row; j++) {
            x[rows[j]] -= a[j * (n + 1) + i] * x[i];
        }
    }

    if (rank == 0) {
        // Корректировка x_0
        x[0] /= a[row * (n + 1)];
    }
    MPI_Bcast(x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Все процессы содержат корректных вектор x_n решений, можно выводить результаты и завершать программу
    t = MPI_Wtime() - t;

    if (rank == 0) {
        printf("Gaussian Elimination (MPI): n %d, procs %d, time (sec) %.6f\n", n, commsize, t);

        printf("MPI X[%d]: ", n);
        for (int i = 0; i < n; i++) {
            printf("%f ", x[i]);
        }
        printf("\n");
    }

    free(tmp);
    free(rows);
    free(a);
    free(x);
    MPI_Finalize();
    return 0;
}

