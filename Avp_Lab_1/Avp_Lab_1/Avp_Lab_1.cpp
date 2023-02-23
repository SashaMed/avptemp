#include <iostream>
#include <ctime>
#include <stdio.h>
#include <chrono>
#include <immintrin.h>
#include<intrin.h>
#include <emmintrin.h>



using namespace std;
using namespace chrono;

#define LEN 300
#define LEN2 16
#define DMIN -10000000
#define DMAX 10000000


double**** arr1;
double**** arr2;
double**** res1;
double**** res2;



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


void matrixInit(double**** arr1, double**** arr2)
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




void matrixMulWithSSE(double**** arr1, double**** arr2, double**** res)
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
                        auto t1 = _mm_cvtsd_f64(suma);
                        auto t2 = _mm_cvtsd_f64(_mm_unpackhi_pd(suma, suma));
                        res[i][j][m][n] += t1 + t2;
                    }
                }
            }
        }
    }
    // for (int i = 0; i < LEN; i++)
    // {
    //     for (int j = 0; j < LEN; j++)
    //     {
    //         for (int q = 0; q < LEN2; q++)
    //         {
    //             for (int k = 0; k < LEN2; k++)
    //             {
    //                     auto rowa = _mm256_load_pd(&arr1[i][j][q][0]);
    //                     auto columna = _mm256_set_pd(arr2[j][i][3][k], arr2[j][i][2][k], arr2[j][i][1][k], arr2[j][i][0][k]);
    //                     auto mula = _mm256_mul_pd(rowa, columna);
    //                     auto lowa = _mm256_extractf128_pd(mula, 0);
    //                     auto higha = _mm256_extractf128_pd(mula, 1);
    //                     auto suma = _mm_add_pd(lowa, higha);
    //                     res[i][j][q][k] += suma[0] + suma[1];
    //             }
    //         }
    //     }
    // }
    high_resolution_clock::time_point chronoEnd = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(chronoEnd - chronoStart);
    printf("with avx vec %f seconds\n", time_span.count());
}



void printMatrix(double** res)
{
    for (int i = 0; i < LEN2; i++)
    {
        for (int j = 0; j < LEN2; j++)
        {
            printf("%10f ", res[i][j]);
        }
        printf("\n");
    }
}


//optimize("no-unroll-loops")
void matrixMulWithoutVectorization(double**** arr1, double**** arr2, double**** res)
{
    high_resolution_clock::time_point chronoStart = high_resolution_clock::now();
    //#pragma loop(no_vector)
    for (int i = 0; i < LEN; i++) {
        for (int j = 0; j < LEN; j++) {
            for (int k = 0; k < LEN; k++) {
                for (int m = 0; m < LEN2; m++) {
                    for (int n = 0; n < LEN2; n++) {
                        for (int p = 0; p < LEN2; p++) {
                            res[i][j][m][n] += arr1[i][k][m][p] * arr2[k][j][p][n];
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




//__attribute__((target("sse,avx")))
void mulSubMatrix(double** __restrict arr1, double** __restrict arr2, double** __restrict res)
{
    const double** __restrict a1 = (const double** __restrict)arr1, ** __restrict a2 = (const double** __restrict)arr2;
    double ** __restrict r = res;
    //for (int row = 0; row < LEN2; ++row)
    //{
    //    double t = 0;
    //    for (int col = 0; col < 4 * LEN2; col++)
    //    {
    //        r[row][col % 4] += a1[row][col / 4] * a2[col / 4][col % 4];
    //    }
    //}
     for (int row = 0; row < LEN2; ++row) 
     {
         for (int i = 0; i < LEN2; ++i) 
         {
             double a_row_i = a1[row][i];
             for (int col = 0; col < LEN2; ++col) 
             {
                 r[row][col] += a_row_i * a2[i][col];
             }
         }
     }
}


//__attribute__((target("sse,avx")))
void  matrixMulWithVectorization(double**** arr1, double**** arr2, double**** res)
{
    high_resolution_clock::time_point chronoStart = high_resolution_clock::now();
    for (int i = 0; i < LEN; i++)
    {
        for (int j = 0; j < LEN; j++)
        {
            for (int q = 0; q < LEN; q++)
            {
                mulSubMatrix(arr1[i][q], arr2[q][j], res[i][j]);
            }
        }
    }
    high_resolution_clock::time_point chronoEnd = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(chronoEnd - chronoStart);
    printf("with vec %f seconds\n", time_span.count());
}





int main()
{

    arr1 = memoryInit();
    arr2 = memoryInit();
    res1 = memoryInit();
    res2 = memoryInit();
    matrixInit(arr1, arr2);
    cout << "init end\n";
    matrixMulWithVectorization(arr1, arr2, res1);
    matrixMulWithoutVectorization(arr1, arr2, res2);
    matrixMulWithSSE(arr1, arr2, res2);
    //for (int i = 0; i < LEN; i++)
    //{
    //    for (int j = 0; j < LEN; j++)
    //    {
    //        for (int q = 0; q < LEN2; q++)
    //        {
    //            for (int k = 0; k < LEN2; k++)
    //            {
    //                //cout << res1[i][j][q][k] << "  " << res2[i][j][q][k] << endl;
    //                //if (res1[i][j][q][k] != res2[i][j][q][k])
    //                //{
    //                //   
    //                //}
    //            }
    //        }
    //    }
    //}
}


//g++ laba1.cpp -std=c++17 -Ofast -march=native -Wall -mavx -mfma -fopt-info-vec -o laba

//g++ laba1.cpp -std=c++17 -Ofast -march=native -Wall -mavx -mfma -fno-tree-vectorize -o laba_without