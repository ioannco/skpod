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
#define  Min(a, b) ((a)<(b)?(a):(b))
#define  IND(i, j, k) (N * N * (i) + N * (j) + (k))

const float NNN3 = 1.f/(N*N*N);

float maxeps = 0.1e-7f;
int itmax = 100;

void relax_columns(
	float * A,
	float * columns_block,
	int * rank_column_count_map,
	int * rank_column_offset_map,
	int ranks_size,
	int proc_rank,
	MPI_Datatype MPI_FLOAT_COLUMN
);

float relax_planes(
	float * A,
	float * planes_block,
	int * rank_plane_count_map,
	int * rank_plane_offset_map,
	int ranks_size,
	int proc_rank
);

void init(float * A);

void verify(float * A);

void run();

double getclock();

void print(float * A);


int main(int an, char **as) {
	MPI_Init(&an, &as);

	float * A;

	int ranks_size;
	int proc_rank;

	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ranks_size);

	if (proc_rank == 0) {
		A = calloc(sizeof(float), N * N * N);
		init(A);
	}

	int * rank_column_count_map = calloc(sizeof(int), ranks_size);
	int * rank_column_offset_map = calloc(sizeof(int), ranks_size);
	int * rank_plane_count_map = calloc(sizeof(int), ranks_size);
	int * rank_plane_offset_map = calloc(sizeof(int), ranks_size);

	for (int i = 0, capacity = N * N; i < ranks_size; i++) {
		int count = Min(capacity, (N * N + ranks_size) / ranks_size);
		rank_column_count_map[i] = count;
		rank_column_offset_map[i] = N * N - capacity;
		capacity -= count;
	}

	for (int i = 0, capacity = N ; i < ranks_size; i++) {
		int count = Min(capacity, (N  + ranks_size) / ranks_size);
		rank_plane_count_map[i] = count * N * N;
		rank_plane_offset_map[i] = (N - capacity) * N * N;
		capacity -= count;
	}

	float * columns_block = calloc(sizeof(float), rank_column_count_map[proc_rank] * N);
	float * planes_block = calloc(sizeof(float), rank_plane_count_map[proc_rank]);

	MPI_Datatype MPI_FLOAT_COLUMN;
	MPI_Datatype MPI_FLOAT_COLUMN_RESIZED;
	MPI_Type_vector(N, 1, N * N, MPI_FLOAT, &MPI_FLOAT_COLUMN);
	MPI_Type_commit(&MPI_FLOAT_COLUMN);
	MPI_Type_create_resized(MPI_FLOAT_COLUMN, 0, sizeof(float), &MPI_FLOAT_COLUMN_RESIZED);
	MPI_Type_commit(&MPI_FLOAT_COLUMN_RESIZED);

	for (int it = 0; it < itmax; it++) {
		float eps = 0.f;
		relax_columns(A, columns_block, rank_column_count_map, rank_column_offset_map, ranks_size, proc_rank, MPI_FLOAT_COLUMN_RESIZED);
		eps = relax_planes(A, planes_block, rank_plane_count_map, rank_plane_offset_map, ranks_size, proc_rank);
		if (eps < maxeps) break;
	}

	if (proc_rank == 0) {
		verify(A);
		free(A);
	}

	MPI_Type_free(&MPI_FLOAT_COLUMN);
	MPI_Type_free(&MPI_FLOAT_COLUMN_RESIZED);
	MPI_Finalize();
	free(rank_column_count_map);
	free(rank_column_offset_map);
	free(rank_plane_offset_map);
	free(rank_plane_count_map);
	free(columns_block);
	free(planes_block);
}


void init(float * A) {
	for (int i = 0; i <= N - 1; i++)
		for (int j = 0; j <= N - 1; j++)
			for (int k = 0; k <= N - 1; k++) {
				if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1)
					A[IND(i, j, k)] = 0.f;
				else A[IND(i, j, k)] = (4.f + i + j + k);
			}
}

float relax(float * A) {
	float eps = 0;

	for (int i = 2; i <= N - 3; i++)
		for (int j = 1; j <= N - 2; j++)
			for (int k = 1; k <= N - 2; k++) {
				A[IND(i, j, k)] = (A[IND(i - 1, j, k)] + A[IND(i - 2, j, k)] + A[IND(i + 1, j, k)] + A[IND(i + 2, j, k)]) * 0.25f;
			}

	for (int i = 1; i <= N - 2; i++)
		for (int j = 2; j <= N - 3; j++)
			for (int k = 1; k <= N - 2; k++) {
				A[IND(i, j, k)] = (A[IND(i, j - 1, k)] + A[IND(i, j - 2, k)] + A[IND(i, j + 1, k)] + A[IND(i, j + 2, k)]) * 0.25f;
			}

	for (int i = 1; i <= N - 2; i++)
		for (int j = 1; j <= N - 2; j++)
			for (int k = 2; k <= N - 3; k++) {
				float e;
				e = A[IND(i, j, k)];
				A[IND(i, j, k)] = (A[IND(i, j, k - 1)] + A[IND(i, j, k - 2)] + A[IND(i, j, k + 1)] + A[IND(i, j, k + 2)]) * 0.25f;
				eps = Max(eps, fabsf(e - A[IND(i, j, k)]));
			}

	return eps;

}

void relax_columns(
	float * A,
	float * columns_block,
	int * rank_column_count_map,
	int * rank_column_offset_map,
	int ranks_size,
	int proc_rank,
	MPI_Datatype MPI_FLOAT_COLUMN
) {
	int column_count = rank_column_count_map[proc_rank];
	int size = column_count * N;

	MPI_Scatterv(A, rank_column_count_map, rank_column_offset_map, MPI_FLOAT_COLUMN, columns_block, size, MPI_FLOAT, 0, MPI_COMM_WORLD);
	for (int column_index = 0; column_index < column_count; column_index++) {
		for (int i = 2; i < N - 2; i++) {
			columns_block[N * column_index + i] = (
				columns_block[N * column_index + i - 2] +
					columns_block[N * column_index + i - 1] +
					columns_block[N * column_index + i + 1] +
					columns_block[N * column_index + i + 2]
			) * 0.25f;
		}
	}
	MPI_Gatherv(columns_block, size, MPI_FLOAT, A, rank_column_count_map, rank_column_offset_map, MPI_FLOAT_COLUMN, 0, MPI_COMM_WORLD);

}

float relax_planes(
	float * A,
	float * planes_block,
	int * rank_plane_count_map,
	int * rank_plane_offset_map,
	int ranks_size,
	int proc_rank
) {
	float eps = 0.f, local_eps = 0.f;

	int size = rank_plane_count_map[proc_rank];
	int plane_count = size / (N * N);

	MPI_Scatterv(A, rank_plane_count_map, rank_plane_offset_map, MPI_FLOAT, planes_block, size, MPI_FLOAT, 0, MPI_COMM_WORLD);
	for (int i = 0; i < plane_count; i++) {
		for (int j = 2; j < N - 2; j++) {
			for (int k = 0; k < N; k++) {
				planes_block[IND(i, j, k)] = (
					planes_block[IND(i, j - 1, k)] +
						planes_block[IND(i, j - 2, k)] +
						planes_block[IND(i, j + 1, k)] +
						planes_block[IND(i, j + 2, k)]
				) * 0.25f;
			}
		}
	}

	for (int i = 0; i < plane_count; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 2; k < N - 2; k++) {
				float e = planes_block[IND(i, j, k)];
				planes_block[IND(i, j, k)] = (
					planes_block[IND(i, j, k - 1)] +
						planes_block[IND(i, j, k - 2)] +
						planes_block[IND(i, j, k + 1)] +
						planes_block[IND(i, j, k + 2)]
				) * 0.25f;
				local_eps = Max(local_eps, fabsf(e - planes_block[IND(i, j, k)]));
			}
		}
	}
	MPI_Gatherv(planes_block, size, MPI_FLOAT, A, rank_plane_count_map, rank_plane_offset_map, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Reduce(&local_eps, &eps, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Bcast(&eps, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	return eps;
}

void verify(float * A) {
	float s;

	s = 0.;
	for (int i = 0; i <= N - 1; i++)
		for (int j = 0; j <= N - 1; j++)
			for (int k = 0; k <= N - 1; k++) {
				s = s + A[IND(i, j, k)] * (float) (i + 1) * (float) (j + 1) * (float) (k + 1) * NNN3;
			}


	printf("S = %f\n", s);
	printf("CHECKSUM = %f\n", (float) CHECKSUM);
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


void print(float * A) {
	for (int i = 0; i < N; i++) {
		printf("i%2d  ", i);
		for (int header = 0; header < N; header++) {
			printf("k%2d      ", header);
		}
		printf("\n");
		for (int j = 0; j < N; j++) {
			printf("j%2d  ", j);
			for (int k = 0; k < N; k++) {
				printf("%f ", A[IND(i, j, k)]);
			}
			printf("\n");
		}
		printf("\n");
	}
}