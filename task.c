#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>
#include "config.h"

#define  Max(a, b) ((a)>(b)?(a):(b))

#define TASKS_COUNT  NUM_THREADS * 4

const float NNN3 = 1.f / (N * N * N);

float maxeps = 0.1e-7;
int itmax = 100;
int i, j, k, task_offset;

float eps;
float A[N][N][N];


void relax();
void init();
void verify();
void run();
double measure_time(void (*func)());
double getclock();

double getclock()
{
    struct timeval Tp;
    gettimeofday (&Tp, NULL);
    return Tp.tv_sec + Tp.tv_usec * 1.0e-6;
}

int main(int an, char **as) {
    run();
}

void run() {
    int it;
    init();

#pragma omp parallel num_threads(NUM_THREADS)
    {
#pragma omp master
        {
            double tstart = getclock();
            double timer = 0.;

            for (it = 1; it <= itmax; it++) {
                eps = 0.;
                relax();
                if (eps < maxeps) break;
            }

#pragma omp taskwait
        double tend = getclock();
        timer = tend - tstart;

        printf("%f", timer);
        verify();
        }
    }
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
	const int task_batch_size = N / TASKS_COUNT + 1;
#pragma omp task shared(A) firstprivate(i, j, k, task_offset)
	for (task_offset = 1; task_offset < N - 1; task_offset += task_batch_size){int task_end = task_offset + task_batch_size;

		printf("%d\n", omp_get_thread_num());
		if (task_end > N - 1)
			task_end = N - 1;

		for (k = task_offset; k < task_end; k++) {
			for (i = 2; i <= N - 3; i++) {
				for (j = 1; j <= N - 2; j++) {
					A[i][j][k] = (A[i - 1][j][k] + A[i + 1][j][k] + A[i - 2][j][k] + A[i + 2][j][k]) * 0.25;
				}
			}
		}
	}
#pragma omp taskwait

#pragma omp task shared(A) firstprivate(i, j, k, task_offset)
	for (task_offset = 1; task_offset < N - 1; task_offset += task_batch_size){
		int task_end = task_offset + task_batch_size;
		if (task_end > N - 1)
			task_end = N - 1;

		for (k = task_offset; k < task_end; k++) {
			for (i = 1; i <= N - 2; i++) {
				for (j = 2; j <= N - 3; j++) {
					A[i][j][k] = (A[i][j - 1][k] + A[i][j + 1][k] + A[i][j - 2][k] + A[i][j + 2][k]) * 0.25;
				}
			}
		}
	}
#pragma omp taskwait

#pragma omp task shared(A) firstprivate(i, j, k, task_offset)
	for (task_offset = 1; task_offset <= N - 1; task_offset += task_batch_size) {
		int task_end = task_offset + task_batch_size;
		if (task_end > N - 1)
			task_end = N - 1;

		for (i = task_offset; i < task_end; i++) {
			for (j = 1; j <= N - 2; j++) {
				for (k = 2; k <= N - 3; k++) {
					float e;
					e = A[i][j][k];
					A[i][j][k] = (A[i][j][k - 1] + A[i][j][k + 1] + A[i][j][k - 2] + A[i][j][k + 2]) * 0.25;
					eps = Max(eps, fabs(e - A[i][j][k]));
				}
			}
		}
	}
#pragma omp taskwait

}

void verify() {
    float s;

    s = 0.;
    for (i = 0; i <= N - 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= N - 1; k++) {
                s = s + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) * NNN3;
            }

#ifndef CHECKSUM
	printf("checksum == %f\n", s);
#else
	if (s != (float)CHECKSUM) {
		fprintf(stderr, "Warning: checksum failed! %f != %f\n", s, CHECKSUM);
	}
#endif
}
