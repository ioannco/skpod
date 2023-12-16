#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

long long int N;
float   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;

float eps;
float ***A;

void relax();
void init();
void verify();

int main(int an, char **as)
{
    int it;
    int status= strtoll(as[2], 0, 10);
    N = 2*2*2*2*2*2*strtoll(as[1], 0, 10) + 2;
    A = (float***)malloc(N * sizeof(float**));
    for (i = 0; i < N; ++i) {
        A[i] = (float**)malloc(N * sizeof(float*));
        for (j = 0; j < N; j++){
            A[i][j] = (float*)malloc(N * sizeof(float));
        }
    }

    double time=omp_get_wtime();
    init();
#pragma omp parallel num_threads(status)
    {
#pragma omp master
        {
            for(it=1; it<=itmax; it++)
            {
                eps = 0.;
                relax();
                printf( "it=%4i   eps=%f\n", it,eps);
                if (eps < maxeps) break;
            }
#pragma omp taskwait
            verify();
        }
    }

    printf("time=%f\n",omp_get_wtime()-time);
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; j++){
            free(A[i][j]);
        }
        free(A[i]);
    }
    free(A);
    return 0;
}


void init()
{
    for(k=0; k<=N-1; k++)
        for(j=0; j<=N-1; j++)
            for(i=0; i<=N-1; i++)
            {
                if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
                    A[i][j][k]= 0.;
                else A[i][j][k]= ( 4. + i + j + k) ;
            }
}

void relax()
{
    for(k=1; k<=N-2; k++)
#pragma omp task shared(A) firstprivate(i, j, k)
    {
        for(j=1; j<=N-2; j++)
            for(i=2; i<=N-3; i++)
            {
                A[i][j][k] = (A[i-1][j][k]+A[i+1][j][k]+A[i-2][j][k]+A[i+2][j][k])/4.;
            }
    }
#pragma omp taskwait

    for(k=1; k<=N-2; k++)
#pragma omp task shared(A) firstprivate(i, j, k)
    {
        for(i=1; i<=N-2; i++)
            for(j=2; j<=N-3; j++)
            {
                A[i][j][k] =(A[i][j-1][k]+A[i][j+1][k]+A[i][j-2][k]+A[i][j+2][k])/4.;
            }
    }
#pragma omp taskwait

    for(i=1; i<=N-2; i++)
#pragma omp task shared(A, eps) firstprivate(i, j, k)
    {
        float tmp = 0;
        for(j=1; j<=N-2; j++)
            for(k=2; k<=N-3; k++)
            {
                float e;
                e=A[i][j][k];
                A[i][j][k] = (A[i][j][k-1]+A[i][j][k+1]+A[i][j][k-2]+A[i][j][k+2])/4.;
                tmp=Max(tmp,fabs(e-A[i][j][k]));
            }
#pragma omp critical
        {
            eps = Max(eps, tmp);
        }
    }
#pragma omp taskwait
}

void verify()
{
    float s;

    s=0.;
    for(k=0; k<=N-1; k++)
        for(j=0; j<=N-1; j++)
            for(i=0; i<=N-1; i++)
            {
                s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
            }
    printf("  S = %f\n",s);

}
