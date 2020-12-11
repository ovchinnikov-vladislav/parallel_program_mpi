#include <mpi/mpi.h>
#include <stdlib.h>
#include <stdio.h>

/**
 * Получение количества строк матрицы, которые будут зарезервированы под один процесс
 * @param total
 * @param commsize
 * @param rank
 * @return
 */
int get_num_rows(int total, int commsize, int rank) {
    int n = total;
    int q = n / commsize;
    if (n % commsize) {
        q++;
    }
    int r = commsize * q - n;

    int nrows = q;
    if (rank >= commsize - r) {
        nrows = q - 1;
    }
    return nrows;
}

/**
 * Вывод распределения системы уравнений по процессам
 * @param rank
 * @param commsize
 * @param rows
 * @param nrows
 */
void distribution_equations_process(int rank, int commsize, int *rows, int nrows) {
    // Вывод в порядке process 0, 1, 2, ... , p - 1
    MPI_Recv(NULL, 0, MPI_INT, (rank > 0) ? rank - 1 : MPI_PROC_NULL, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Process [%d]: ", rank);
    for (int i = 0; i < nrows; i++) {
        printf("%d ", rows[i]);
    }
    printf("\n");
    MPI_Ssend(NULL, 0, MPI_INT, rank != commsize - 1 ? rank + 1 : MPI_PROC_NULL, 0, MPI_COMM_WORLD);
}

/**
 * Рандомное заполнение числами локальной части матрицы ЗЛП
 * @param a
 * @param n
 * @param rows
 * @param nrows
 * @param rank
 * @param comm_size
 */
void random_init_matrix_local(double *a, int n, int *rows, int nrows, int rank, int comm_size) {
    for (int i = 0; i < nrows; i++) {
        rows[i] = rank + comm_size * i;
        srand(rows[i] * (n + 1));
        for (int j = 0; j < n; j++) {
            a[i * (n + 1) + j] = rand() % 100 + 1;
        }
        a[i * (n + 1) + n] = rand() % 100 + 1;
    }
}

/**
 * Прямой ход решения метода Гаусса
 * @param a
 * @param n
 * @param rows
 * @param nrows
 * @param tmp
 * @param row
 * @param rank
 * @param comm_size
 */
void elimination(double *a, int n, double *x, const int *rows, int nrows, double *tmp, int rank, int comm_size) {
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
            MPI_Bcast(tmp, n + 1, MPI_DOUBLE, i % comm_size, MPI_COMM_WORLD);
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
}

/**
 * Обратный ход решения методом Гаусса
 * @param a
 * @param n
 * @param x
 * @param rows
 * @param nrows
 * @param rank
 * @param comm_size
 */
void back_substitution(double *a, int n, double *x, const int *rows, int nrows, int rank, int comm_size) {
    int row = nrows - 1;
    for (int i = n - 1; i > 0; i--) {
        if (row >= 0) {
            if (i == rows[row]) {
                // Передаем найдненное x_i
                x[i] /= a[row * (n + 1) + i];
                MPI_Bcast(&x[i], 1, MPI_DOUBLE, rank, MPI_COMM_WORLD);
                row--;
            } else {
                MPI_Bcast(&x[i], 1, MPI_DOUBLE, i % comm_size, MPI_COMM_WORLD);
            }
        } else {
            MPI_Bcast(&x[i], 1, MPI_DOUBLE, i % comm_size, MPI_COMM_WORLD);
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
}

// TODO: починить функцию вывода матрицы на экран
void print(double *a, double *x, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - 1; j++) {
            printf("%12.4f ", a[i * (n + 1) + j]);
        }
        printf(" | %12.4f\n\n", a[i * (n + 1) + n]);
    }
    if (x != NULL) {
        for (int i = 0; i < n; i++) {
            printf("\tx_%i = %12.4f\n", i, x[i]);
        }
    }
}

int start_gauss_parall_mpi_2(int argc, char **argv) {
    int n = 10;

    if (argc == 2) {
    	n = atoi(argv[1]);
    }

    int rank, comm_size;
    MPI_Init(&argc, &argv);
    double t = MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int nrows = get_num_rows(n, comm_size, rank);
    int *rows = malloc(sizeof(*rows) * nrows);          // Номера локальных строк

    // Матрица дополнена столбцом для вектора b
    double *a = malloc(sizeof(*a) * nrows * (n + 1));
    double *x = malloc(sizeof(*x) * n);
    double *tmp = malloc(sizeof(*tmp) * (n + 1));

    random_init_matrix_local(a, n, rows, nrows, rank, comm_size);
    distribution_equations_process(rank, comm_size, rows, nrows);
    elimination(a, n, x, rows, nrows, tmp, rank, comm_size);
    back_substitution(a, n, x, rows, nrows, rank, comm_size);

    // Все процессы содержат корректных вектор x_n решений, можно выводить результаты и завершать программу
    t = MPI_Wtime() - t;

    free(tmp);
    free(rows);
    free(a);
    free(x);

    MPI_Finalize();

    if (rank == 0) {
        printf("Information: "
               "\n\tnumber of equations: %d"
               "\n\tnumber of processes: %d"
               "\n\texecution time (seconds): %.6f\n", n, comm_size, t);

        printf("X[%d]: [ ", n);
        for (int i = 0; i < n; i++) {
            printf("x%i = %f ; ", i, x[i]);
        }
        printf("\n ]");
    }

    return 0;
}

