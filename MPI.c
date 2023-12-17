#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <assert.h>
#include <sys/time.h>
#include <mpi.h>
#include "config.h"

#define  Max(a, b) ((a)>(b)?(a):(b))

const float NNN3 = 1.f/(N*N*N);

float maxeps = 0.1e-7f;
int itmax = 100;
int i, j, k;

float eps;
float A[N][N][N];

void relax();
void init();
void verify();
void run();
double getclock();

int main(int an, char **as) {

    int it;

    init();

    double timer = 0.;
    int cnt = 0;

    for (it = 1; it <= itmax; it++) {
        ++cnt;
        eps = 0.;
        double bench_t_start = getclock();
        relax();
        double bench_t_end = getclock();
        timer += bench_t_end - bench_t_start;
        if (eps < maxeps) break;
    }
    printf("%f\n", timer);

    verify();
}

void init() {
    for (i = 0; i <= N - 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= N - 1; k++) {
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1)
                    A[i][j][k] = 0.f;
                else A[i][j][k] = (4.f + i + j + k);
            }
}

void relax() {
    for (i = 2; i <= N - 3; i++)
        for (j = 1; j <= N - 2; j++)
            for (k = 1; k <= N - 2; k++) {
                A[i][j][k] = (A[i - 1][j][k] + A[i + 1][j][k] + A[i - 2][j][k] + A[i + 2][j][k]) * 0.25f;
            }

    for (i = 1; i <= N - 2; i++)
        for (j = 2; j <= N - 3; j++)
            for (k = 1; k <= N - 2; k++) {
                A[i][j][k] = (A[i][j - 1][k] + A[i][j + 1][k] + A[i][j - 2][k] + A[i][j + 2][k]) * 0.25f;
            }

    for (i = 1; i <= N - 2; i++)
        for (j = 1; j <= N - 2; j++)
            for (k = 2; k <= N - 3; k++) {
                float e;
                e = A[i][j][k];
                A[i][j][k] = (A[i][j][k - 1] + A[i][j][k + 1] + A[i][j][k - 2] + A[i][j][k + 2]) * 0.25f;
                eps = Max(eps, fabsf(e - A[i][j][k]));
            }

}

void verify() {
    float s;

    s = 0.;
    for (i = 0; i <= N - 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= N - 1; k++) {
                s = s + A[i][j][k] * (float) (i + 1) * (float) (j + 1) * (float) (k + 1) * NNN3;
            }

#ifndef CHECKSUM
    printf("  S = %f\n", s);
#else
    assert (s == (float) CHECKSUM);
#endif
}

double getclock()
{
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, NULL);
  if (stat != 0) {
	printf ("Error return from gettimeofday: %d", stat);
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}