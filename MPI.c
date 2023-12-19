#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <assert.h>
#include "config.h"

#define  Max(a, b) ((a)>(b)?(a):(b))
#define  Min(a, b) ((a)<(b)?(a):(b))

#define ROOT_RANK (!rank)

const float NNN3 = 1.f / (N * N * N);

double maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
double timer;

float eps;
float A[N][N][N];
int ranks_count, rank, step, block_start_idx, block_end_idx;

void relax();

void init();

void verify();

int main(int argc, char *argv[]) {
    int it;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (ROOT_RANK) {
        init();
    }

    MPI_Bcast(A, N * N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    step = (N + ranks_count - 3) / ranks_count;
    block_start_idx = step * rank + 1;
    block_end_idx = Min(N - 2, block_start_idx + step - 1);

    timer = MPI_Wtime();
    for (it = 1; it <= itmax; it++) {
        eps = 0.;
        relax();
        if (ROOT_RANK) {
            if (eps < maxeps) break;
        }
    }

    timer = MPI_Wtime() - timer;
    if (!rank) {
        printf("%f\n", timer);
        //verify();
    }

    MPI_Finalize();
    return 0;
}


void init() {
    for (i = 0; i <= N - 1; i++) {
        for (j = 0; j <= N - 1; j++) {
            for (k = 0; k <= N - 1; k++) {
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1)
                    A[i][j][k] = 0.;
                else A[i][j][k] = (4. + i + j + k);
            }
        }
    }
}

void relax() {
    for (i = 2; i <= N - 3; i++) {
        for (j = block_start_idx; j <= block_end_idx; j++) {
            for (k = 1; k <= N - 2; k++) {
                A[i][j][k] = (A[i - 1][j][k] + A[i + 1][j][k] + A[i - 2][j][k] + A[i + 2][j][k]) * 0.25;
            }
        }
    }

    for (int proc = 0; proc < ranks_count; ++proc) {
        int proc_block_start_idx = step * proc + 1;
        int proc_block_end_idx = Min(N - 2, proc_block_start_idx + step - 1);
        int size = proc_block_end_idx - proc_block_start_idx + 1;
        for (i = 2; i <= N - 3; ++i) {
            if (size > 0) {
                MPI_Bcast(&A[i][proc_block_start_idx][0], size * N, MPI_FLOAT, proc, MPI_COMM_WORLD);
            }
        }
    }

    for (i = block_start_idx; i <= block_end_idx; i++) {
        for (j = 2; j <= N - 3; j++) {
            for (k = 1; k <= N - 2; k++) {
                A[i][j][k] = (A[i][j - 1][k] + A[i][j + 1][k] + A[i][j - 2][k] + A[i][j + 2][k]) * 0.25;
            }
        }
    }

    float proc_eps = 0.0;
    for (i = block_start_idx; i <= block_end_idx; i++) {
        for (j = 1; j <= N - 2; j++) {
            for (k = 2; k <= N - 3; k++) {
                double e;
                e = A[i][j][k];
                A[i][j][k] = (A[i][j][k - 1] + A[i][j][k + 1] + A[i][j][k - 2] + A[i][j][k + 2]) * 0.25;
                proc_eps = Max(proc_eps, fabs(e - A[i][j][k]));
            }
        }
    }

    for (int proc = 0; proc < ranks_count; ++proc) {
        int proc_block_start_idx = step * proc + 1;
        int proc_block_end_idx = Min(N - 2, proc_block_start_idx + step - 1);
        int size = proc_block_end_idx - proc_block_start_idx + 1;
        if (size > 0) {
            MPI_Bcast(&A[proc_block_start_idx][0][0], size * N * N, MPI_FLOAT, proc, MPI_COMM_WORLD);
        }
    }

    MPI_Allreduce(&proc_eps, &eps, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
}

void verify() {
    float s;

    s = 0.;
    for (i = 0; i <= N - 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= N - 1; k++) {
                s = s + A[i][j][k] * (float) (i + 1) * (float) (j + 1) * (float) (k + 1) * NNN3;
            }
}