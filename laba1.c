//#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Optimization flags
//#pragma GCC option("arch=native","tune=native","no-zero-upper") //Enable AVX
//#pragma GCC target("avx")  //Enable AVX
//#include <x86intrin.h> //AVX/SSE Extensions
//#include <bits/stdc++.h> //All main STD libraries
//#include <iostream>
//#include <ctime>
#include <stdio.h>
#include <stdlib.h>
//#include <Windows.h>
//#include <chrono>
//#include <immintrin.h>

//using namespace std;
//using namespace chrono;

#define LEN 1000
#define LEN2 4
#define DMIN -10000000
#define DMAX 10000000






double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


double**** memoryInit()
{
    double**** arr1 = NULL;
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


void matrixInit(double ****arr1, double ****arr2)
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


void matrixMulWithoutVectorization(double**** arr1, double**** arr2, double**** res)
{


    //high_resolution_clock::time_point chronoStart = high_resolution_clock::now();

    #pragma loop(no_vector)
    //#pragma novector
    for (int i = 0; i < LEN; i++)
    {
        //#pragma loop(no_vector) 
        for (int j = 0; j < LEN; j++)
        {
            //#pragma loop(no_vector)
            for (int q = 0; q < LEN2; q++)
            {
                //#pragma loop(no_vector)
                for (int k = 0; k < LEN2; k++)
                {
                    double mulRes = 0;
                    //#pragma loop(no_vector)
                    for (int x = 0; x < LEN2; x++)
                    {
                        //#pragma loop(no_vector)
                        for (int y = 0; y < LEN2; y++)
                        {
                            mulRes += arr1[i][j][x][y] * arr2[i][j][y][x];
                        }
                    }
                    res[i][j][q][k] = mulRes;
                }
            }
        }
    }
    //high_resolution_clock::time_point chronoEnd = high_resolution_clock::now();
    //duration<double> time_span = duration_cast<duration<double>>(chronoEnd - chronoStart);
    //printf("without vec %f secoonds\n", time_span.count());
    printf("without vec  secoonds\n");
}



void matrixMulWithVectorization(double**** __restrict arr1, double**** __restrict arr2, double**** __restrict res)
{


    //high_resolution_clock::time_point chronoStart = high_resolution_clock::now();
    for (int i = 0; i < LEN; i++)
    {
        for (int j = 0; j < LEN; j++)
        {
            for (int q = 0; q < LEN2; q++)
            {
                for (int k = 0; k < LEN2; k++)
                {
                    for (int x = 0; x < LEN2; x++)
                    {
                        //auto temp = 0;
                        for (int y = 0; y < LEN2; y++)
                        {
                            res[i][j][q][k] += arr1[i][j][x][y] * arr2[i][j][y][x];
                        }
                        //res[i][j][q][k] = temp;
                    }
                }
            }
        }
    }
    //high_resolution_clock::time_point chronoEnd = high_resolution_clock::now();
    //duration<double> time_span = duration_cast<duration<double>>(chronoEnd - chronoStart);
    //printf("with vec %f secoonds\n", time_span.count());
    printf("with vec secoonds\n");
}

int main()
{   
    double**** __restrict arr1 = memoryInit();
    double**** __restrict arr2 = memoryInit();
    double**** __restrict res1 = memoryInit();
    double**** __restrict res2 = memoryInit();
    matrixInit(arr1, arr2);
    printf("init is over");
    matrixMulWithVectorization(arr1, arr2, res1);
    matrixMulWithoutVectorization(arr1, arr2, res2);

}


