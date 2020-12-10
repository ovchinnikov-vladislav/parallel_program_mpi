#include <stdlib.h>
#include <stdio.h>
#include "gauss.h"

// Вывод на экран системы уравнений в виде матрицы
void print(double *a, double *b, double *x, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%12.4f ", a[i * n + j]);
        }
        printf(" | %12.4f\n\n", b[i]);
    }
    if (x != NULL) {
        for (int i = 0; i < n; i++) {
            printf("\tx_%i = %12.4f\n", i, x[i]);
        }
    }
}

int start_gauss(int argc, char **argv) {
    int n = 10;
    double *a = malloc(sizeof(*a) * n * n); // Матрица коэффициентов
    double *b = malloc(sizeof(*b) * n);     // Столбец свободных членов
    double *x = malloc(sizeof(*x) * n);     // Незвестные

    // Инициализация системы уравнений рандомными значениями
    for (int i = 0; i < n; i++) {
        srand(i * (n + 1));
        for (int j = 0; j < n; j++) {
            a[i * n + j] = rand() % 100 + 1;
        }
        b[i] = rand() % 100 + 1;
    }

    print(a, b, NULL, n);

    // Прямой ход решения
    for (int k = 0; k < n - 1; k++) {
        // Исключение x_i из строк k+1...n-1
        double pivot = a[k * n + k];
        for (int i = k + 1; i < n; i++) {
            // Из уравнения (строки) i вычитается уравнение k
            double lik = a[i * n + k] / pivot;
            for (int j = k; j < n; j++) {
                a[i * n + j] -= lik * a[k * n + j];
            }
            b[i] -= lik * b[k];
        }
    }

    // Обратный ход решения
    for (int k = n - 1; k >= 0; k--) {
        x[k] = b[k];
        for (int i = k + 1; i < n; i++) {
            x[k] -= a[k * n + i] * x[i];
        }
        x[k] /= a[k * n + k];
    }

    print(a, b, x, n);


    return 0;
}


