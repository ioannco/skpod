#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

void init(int n, float * A)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
            {
                if (i == 0 || i == n - 1 || j == 0 || j == n - 1 || k == 0 || k == n - 1)
                    A[i * n * n + n * j + k] = 0.;
                else A[i * n * n + n * j + k] = (4. + i + j + k) ;
            }
}

float relax(int n, float * A)
{
    float eps = 0.;

    for (int k = 1; k < n - 1; k++)
        for (int j = 1; j < n - 1; j++)
            for (int i = 1; i < n - 1; i++)
            {
                A[i * n * n + n * j + k] = (A[(i - 1) * n * n + n * j + k] + A[(i + 1) * n * n + n * j + k]) / 2.;
            }

    for (int k = 1; k < n - 1; k++)
        for (int j = 1; j < n - 1; j++)
            for (int i = 1; i < n - 1; i++)
            {
                A[i * n * n + n * j + k] = (A[i * n * n + n * (j - 1) + k] + A[i * n * n + n * (j + 1) + k]) / 2.;
            }

    for (int k = 1; k < n - 1; k++)
        for (int j = 1; j < n - 1; j++)
            for (int i = 1; i < n - 1; i++)
            {
                float e;
                e = A[i * n * n + n * j + k];
                A[i * n * n + n * j + k] = (A[i * n * n + n * j - 1 + k] + A[i * n * n + n * j + 1 + k]) / 2.;
                eps = Max(eps, fabs(e - A[i * n * n + n * j + k]));
            }

    return eps;
}

float relax_parallel(int n, float * A, int myrank, int size, float * my_A1, float * my_A2, int * counts1, int * displc1, int * counts2, int * displc2, MPI_Datatype COLRES)
{
    float eps = 0., eps_gl = 0.;
    int n2 = n * n;

    int my_n = counts1[myrank];
    int arr_size = my_n * n;

    MPI_Scatterv(A, counts1, displc1, COLRES, my_A1, arr_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int count = 0;
    for (int jk = 0; jk < my_n; jk++) {
        for (int i = 0; i < n; i++) {
            if (i != 0 && i != n - 1) {
                my_A1[count] = (my_A1[count - 1] + my_A1[count + 1]) / 2.; //i-coord cycle
            }
            count++;
        }
    }

    MPI_Gatherv(my_A1, arr_size, MPI_FLOAT, A, counts1, displc1, COLRES, 0, MPI_COMM_WORLD);

    //---------------------------------------------------------

    arr_size = counts2[myrank];
    my_n = arr_size / n2;

    MPI_Scatterv(A, counts2, displc2, MPI_FLOAT, my_A2, arr_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    count = 0;
    for (int i = 0; i < my_n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                if (j != 0 && j != n - 1) {
                    my_A2[count] = (my_A2[count - n] + my_A2[count + n]) / 2.; //j-coord cycle
                }
                count++;
            }
        }
    }

    count = 0;
    for (int i = 0; i < my_n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                if (k != 0 && k != n - 1) {
                    float e = my_A2[count];
                    my_A2[count] = (my_A2[count - 1] + my_A2[count + 1]) / 2.; //k-coord cycle
                    eps = Max(eps, fabs(e - my_A2[count]));
                }
                count++;
            }
        }
    }

    MPI_Gatherv(my_A2, arr_size, MPI_FLOAT, A, counts2, displc2, MPI_FLOAT, 0, MPI_COMM_WORLD);
//    MPI_Reduce(&eps, &eps_gl, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
//    MPI_Bcast(&eps_gl, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Allerduce(&eps)

    return eps_gl;
}

void wrapper(int n, float * A, int myrank, int size, int itmax, float mineps) {
    int n2 = n * n;

    int * counts1, * displc1;
    counts1 = calloc(size, sizeof(*counts1));
    displc1 = calloc(size, sizeof(*displc1));

    counts1[0] = (n2 / size + (0 < n2 % size));
    displc1[0] = 0;
    for (int i = 1; i < size; i++) {
        counts1[i] = counts1[i - 1];
        if (i == n2 % size) counts1[i]--;
        displc1[i] = displc1[i - 1] + counts1[i - 1];
    }

    int * counts2, * displc2;
    counts2 = calloc(size, sizeof(*counts2));
    displc2 = calloc(size, sizeof(*displc2));

    counts2[0] = (n / size + (0 < n % size)) * n2;
    displc2[0] = 0;
    for (int i = 1; i < size; i++) {
        counts2[i] = counts2[i - 1];
        if (i == n % size) counts2[i] -= n2;
        displc2[i] = displc2[i - 1] + counts2[i - 1];
    }

    int arr_size = counts1[myrank] * n;
    float * my_A1 = calloc(arr_size, sizeof(*my_A1));

    arr_size = counts2[myrank];
    float * my_A2 = calloc(arr_size, sizeof(*my_A2));

    MPI_Datatype COL, COLRES;
    MPI_Type_vector(n, 1, n2, MPI_FLOAT, &COL);
    MPI_Type_commit(&COL);
    MPI_Type_create_resized(COL, 0, sizeof(*my_A1), &COLRES);
    MPI_Type_commit(&COLRES);

    for(int it = 0; it < itmax; it++)
    {
        float eps = 0.;
        eps = relax_parallel(n, A, myrank, size, my_A1, my_A2, counts1, displc1, counts2, displc2, COLRES);
        if (eps < mineps) break;
    }

    MPI_Type_free(&COL);
    MPI_Type_free(&COLRES);

    free(my_A1);
    free(my_A2);
    free(counts1);
    free(displc1);
    free(counts2);
    free(displc2);
}

void verify(int n, float * A)
{
    float s = 0.;

    for (int i = 1; i < n - 1; i++)
        for (int j = 1; j < n - 1; j++)
            for (int k = 1; k < n - 1; k++)
            {
                s += A[i * n * n + n * j + k] * (i + 1) * (j + 1) * (k + 1) / (n * n * n);
            }
    printf("S = %f\n", s);

}

int main (int argc, char * argv[])
{
    MPI_Init(&argc, &argv);

    const float mineps = 0.1e-7;
    const int itmax = 100;

    int n = strtol(argv[1], NULL, 10);

    double start, end;

    int myrank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float * A;
    if (myrank == 0) {
        A = calloc(n * n * n, sizeof(*A));
        /*
        init(n, A);
        
        for(int it = 0; it < itmax; it++)
	    {
		    float eps = 0.;
		    eps = relax(n, A);
		    if (eps < mineps) break;
	    }
        
        verify(n, A);
        */
        init(n, A);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (myrank == 0) {
        start = MPI_Wtime();
    }

    wrapper(n, A, myrank, size, itmax, mineps);

    MPI_Barrier(MPI_COMM_WORLD);

    if (myrank == 0) {
        end = MPI_Wtime();
        FILE * f = fopen(argv[2], "w");
        fprintf(f, "%lf\n", end - start);
        fclose(f);
        //verify (n, A);
        free(A);
    }

    MPI_Finalize();

    return 0;
}