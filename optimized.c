#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <assert.h>

#define  Max(a, b) ((a)>(b)?(a):(b))

#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#define MINI_DATASET
#endif

#ifdef MINI_DATASET
#define N (2*2*2*2*2*2 + 2)
#ifndef CHECKSUM
#define CHECKSUM 2723874.750000
#endif
#endif

#ifdef SMALL_DATASET
#define N (2*2*2*2*2*2*2 + 2)
#ifndef CHECKSUM
#define CHECKSUM 54117800.000000
#endif
#endif

#ifdef MEDIUM_DATASET
#define N (2*2*2*2*2*2*2*2 + 2)
#ifndef CHECKSUM
#define CHECKSUM 953108160.000000
#endif
#endif

#ifdef LARGE_DATASET
#define  N  (2*2*2*2*2*2*2*2*2 + 2)
#ifndef CHECKSUM
#define CHECKSUM 12370568192.000000
#endif
#endif

float maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;

float eps;
float A[N][N][N];

void relax();
void init();
void verify();
void run();

int main(int an, char **as) {
    run();
}

void run() {
    int it;

    init();

    for (it = 1; it <= itmax; it++) {
        eps = 0.;
        relax();
        printf("it=%4i   eps=%f\n", it, eps);
        if (eps < maxeps) break;
    }

    verify();
}

void init() {
    for (i = 0; i <= N - 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= N - 1; k++) {
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1)
                    A[i][j][k] = 0.;
                else A[i][j][k] = (4. + i + j + k);
            }
}

void relax() {
    for (i = 2; i <= N - 3; i++)
        for (j = 1; j <= N - 2; j++)
            for (k = 1; k <= N - 2; k++) {
                A[i][j][k] = (A[i - 1][j][k] + A[i + 1][j][k] + A[i - 2][j][k] + A[i + 2][j][k]) / 4.;
            }

    for (i = 1; i <= N - 2; i++)
        for (j = 2; j <= N - 3; j++)
            for (k = 1; k <= N - 2; k++) {
                A[i][j][k] = (A[i][j - 1][k] + A[i][j + 1][k] + A[i][j - 2][k] + A[i][j + 2][k]) / 4.;
            }

    for (i = 1; i <= N - 2; i++)
        for (j = 1; j <= N - 2; j++)
            for (k = 2; k <= N - 3; k++) {
                float e;
                e = A[i][j][k];
                A[i][j][k] = (A[i][j][k - 1] + A[i][j][k + 1] + A[i][j][k - 2] + A[i][j][k + 2]) / 4.;
                eps = Max(eps, fabs(e - A[i][j][k]));
            }

}

void verify() {
    float s;

    s = 0.;
    for (i = 0; i <= N - 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= N - 1; k++) {
                s = s + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N * N * N);
            }

#ifndef CHECKSUM
    printf("  S = %f\n", s);
#else
    assert (s == CHECKSUM);
#endif
}

double measure_time(void(*func)()) {
    clock_t start = clock();
    func();
    clock_t end = clock();
    return ((double) (end - start)) / CLOCKS_PER_SEC;
}