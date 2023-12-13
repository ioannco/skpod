#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define  Max(a, b) ((a)>(b)?(a):(b))

#define  N   ((1 << 6) + 2)
float maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;

float eps;
float A[N][N][N];

void
relax();

void
init();

void
verify();

void
test();

int
run_task(int an, char** as);

int
main(int an, char** as)
{
	run_task(an, as);
}

int
run_task(int an, char** as)
{
	int it;

	init();

	for (it = 1; it <= itmax; it++)
	{
		eps = 0.;
		relax();
		printf("it=%4i   eps=%f\n", it, eps);
		if (eps < maxeps) break;
	}

	verify();

	return 0;
}

void
init()
{
	for (i = 0; i <= N - 1; i++)
		for (j = 0; j <= N - 1; j++)
			for (k = 0; k <= N - 1; k++)
			{
				if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1)
					A[i][j][k] = 0.;
				else A[i][j][k] = (4. + i + j + k);
			}
}

void
relax()
{
#pragma omp parallel for shared(A) private(i, j, k)
	for (i = 2; i <= N - 3; i++)
		for (j = 1; j <= N - 2; j++)
			for (k = 1; k <= N - 2; k++)
			{
				A[i][j][k] = (A[i - 1][j][k] + A[i + 1][j][k] + A[i - 2][j][k] + A[i + 2][j][k]) / 4.;
			}

#pragma omp parallel for shared(A) private(i, j, k)
	for (i = 1; i <= N - 2; i++)
		for (j = 2; j <= N - 3; j++)
			for (k = 1; k <= N - 2; k++)
			{
				A[i][j][k] = (A[i][j - 1][k] + A[i][j + 1][k] + A[i][j - 2][k] + A[i][j + 2][k]) / 4.;
			}

#pragma omp parallel for shared(A) private(i, j, k) reduction(max:eps)
	for (i = 1; i <= N - 2; i++)
		for (j = 1; j <= N - 2; j++)
			for (k = 2; k <= N - 3; k++)
			{
				float e;
				e = A[i][j][k];
				A[i][j][k] = (A[i][j][k - 1] + A[i][j][k + 1] + A[i][j][k - 2] + A[i][j][k + 2]) / 4.;
				eps = Max(eps, fabs(e - A[i][j][k]));
			}

}

void
verify()
{
	float s;

	s = 0.;
	for (i = 0; i <= N - 1; i++)
		for (j = 0; j <= N - 1; j++)
			for (k = 0; k <= N - 1; k++)
			{
				s = s + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N * N * N);
			}

	printf("  S = %f\n", s);
}

void
test()
{
#pragma omp parallel
	{
		printf("%d", omp_get_thread_num());
	}
}
