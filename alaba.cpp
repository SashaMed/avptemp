#include <iostream>
#include <ctime>
#include <stdio.h>
#include <chrono>
#include <immintrin.h>




using namespace std;
using namespace chrono;

#define LEN 1000
#define LEN2 4
#define DMIN -10000000
#define DMAX 10000000


double  arr1[LEN][LEN][LEN2][LEN2] __attribute__((aligned(32)));
double  arr2[LEN][LEN][LEN2][LEN2] __attribute__((aligned(32)));
double  res1[LEN][LEN][LEN2][LEN2] __attribute__((aligned(32)));
double  res2[LEN][LEN][LEN2][LEN2] __attribute__((aligned(32)));

    //double****  __attribute__((aligned (32))) arr2 = memoryInit();
    //double****  __attribute__((aligned (32))) res1 = memoryInit();
    //double****  __attribute__((aligned (32))) res2 = memoryInit();




double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


double**** memoryInit()
{
    double**** arr1 = nullptr;
    arr1 = (double****)calloc(LEN, sizeof(double***));
    for (int i = 0; i < LEN; i++)
    {
        arr1[i] = (double***)calloc(LEN, sizeof(double**));
        for (int j = 0; j < LEN; j++)
        {
            arr1[i][j] = (double**)calloc(LEN2, sizeof(double*));
            for (int q = 0; q < LEN2; q++)
            {
                arr1[i][j][q] = (double*)calloc(LEN2, sizeof(double));
            }
        }
    }
    return arr1;
}


void matrixInit()
{
    for (int i = 0; i < LEN; i++)
    {
        for (int j = 0; j < LEN; j++)
        {
            for (int q = 0; q < LEN2; q++)
            {
                for (int k = 0; k < LEN2; k++)
                {
                    arr1[i][j][q][k] = fRand(DMIN, DMAX);
                    arr2[i][j][q][k] = fRand(DMIN, DMAX);
                }
            }
        }
    }
}




void matrixMulWithSSE()
{
    high_resolution_clock::time_point chronoStart = high_resolution_clock::now();
    for (auto i = 0; i < LEN; i++) {
        for (auto j = 0; j < LEN; j++) {
            for (auto k = 0; k < LEN; k++) {
                for (auto m = 0; m < LEN2; m++) {
                    for (auto n = 0; n < LEN2; n++) 
                    {
                        auto rowa = _mm256_load_pd(&arr1[i][k][m][0]);
                        auto columna = _mm256_set_pd(arr2[k][j][3][n], arr2[k][j][2][n], arr2[k][j][1][n], arr2[k][j][0][n]);
                        auto mula = _mm256_mul_pd(rowa, columna);
                        auto lowa = _mm256_extractf128_pd(mula, 0);
                        auto higha = _mm256_extractf128_pd(mula, 1);
                        auto suma = _mm_add_pd(lowa, higha);               
                        res1[i][j][m][n] += suma[0] + suma[1];
                    }
                }
            }
        }
    }
    high_resolution_clock::time_point chronoEnd = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(chronoEnd - chronoStart);
    printf("with avx vec %f seconds\n", time_span.count());
}



void printMatrix(double **res)
{
    for (int i = 0; i < LEN2; i++)
    {
        for (int j = 0; j <LEN2; j++)
        {
            printf("%10f ", res[i][j]);
        }
        printf("\n");
    }
}



void matrixMulWithoutVectorization()
{
    high_resolution_clock::time_point chronoStart = high_resolution_clock::now();
    for (int i = 0; i < LEN; i++) {
        for (int j = 0; j < LEN; j++) {
            for (int k = 0; k < LEN; k++) {
                #pragma clang loop vectorize(disable) interleave(disable)
                for (int m = 0; m < LEN2; m++) {
                    for (int n = 0; n < LEN2; n++) {
                        for (int p = 0; p < LEN2; p++) {
                            res2[i][j][m][n] += arr1[i][k][m][p] * arr2[k][j][p][n];
                        }
                    }
                }
            }
        }
    }
    high_resolution_clock::time_point chronoEnd = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(chronoEnd - chronoStart);
    printf("without vec %f seconds\n", time_span.count());
}




__attribute__((target("sse,avx")))
void mulSubMatrix(double** arr1, double** arr2, double** res)
{
    for (int i = 0; i < LEN2; i++)
    {
        for (int j = 0; j <LEN2; j++)
        {
            for (int q = 0; q < LEN2; q++)
            {
                res[i][j] += arr1[i][q] * arr2[q][j];
            }   
        }
    }
    //#pragma clang loop vectorize(enable) interleave(enable)
    // for (int row = 0; row < LEN2; ++row) 
    // {
    //     for (int i = 0; i < LEN2; ++i) 
    //     {
    //         int a_row_i = arr1[row][i];
    //         for (int col = 0; col < LEN2; ++col) 
    //         {
    //             res[row][col] += a_row_i * arr2[i][col];
    //         }
    //     }
    // }
}


__attribute__((target("sse,avx")))
void  matrixMulWithVectorization()
{
    high_resolution_clock::time_point chronoStart = high_resolution_clock::now();
    for (int i = 0; i < LEN; i++) {
        for (int j = 0; j < LEN; j++) {
            for (int k = 0; k < LEN; k++) {
                #pragma clang loop vectorize(enable) interleave(enable)
                for (int m = 0; m < LEN2; m++) {
                    for (int n = 0; n < LEN2; n++) {
                        for (int p = 0; p < LEN2; p++) {
                            res1[i][j][m][n] += arr1[i][k][m][p] * arr2[k][j][p][n];
                        }
                    }
                }
            }
        }
    }
    high_resolution_clock::time_point chronoEnd = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(chronoEnd - chronoStart);
    printf("with vec %f seconds\n", time_span.count());
}





int main()
{   
    matrixInit();
    cout << "init end\n";
    matrixMulWithVectorization();
    matrixMulWithoutVectorization();
    matrixMulWithSSE();
    for (int i = 0; i < LEN; i++)
    {
        for (int j = 0; j < LEN; j++)
        {
            for (int q = 0; q < LEN2; q++)
            {
                for (int k = 0; k < LEN2; k++)
                {
                    if (res1[i][j][q][k] != res2[i][j][q][k])
                    {
                        //cout << "i " << i << " j " << j << " q " << q << " k " << k << "   "
                        // << res1[i][j][q][k] << "  " << res2[i][j][q][k] << endl;
                    }
                }
            }
        }
    }
}


//g++ laba1.cpp -std=c++17 -Ofast -march=native -Wall -mavx -mfma -fopt-info-vec -o laba

//g++ laba1.cpp -std=c++17 -Ofast -march=native -Wall -mavx -mfma -fno-tree-vectorize -o laba_without