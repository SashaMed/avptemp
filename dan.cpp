#include <iostream>
#include <random>
#include <chrono>
#include <immintrin.h>
#include <assert.h>
#include <x86intrin.h>

// const int M = 140;
// const int N = 160;
// const int P = 180;
// const int K = 4;


const int M = 300;
const int N = 300;
const int P = 300;
const int K = 4;

void fillFirstMatrix(double mat[][N][K][K], int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int m = 0; m < K; m++) {
                for (int n = 0; n < K; n++) {
                    mat[i][j][m][n] = dis(gen);
                }
            }
        }
    }
}

void fillSecondMatrix(double mat[][P][K][K], int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int m = 0; m < K; m++) {
                for (int n = 0; n < K; n++) {
                    mat[i][j][m][n] = dis(gen);
                }
            }
        }
    }
}

__attribute__((target("sse,avx")))
void multiplyMatrices(double matA[][N][K][K], int rowsA, int colsA, double matB[][P][K][K], int rowsB, int colsB, double matC[][P][K][K]) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                for (int m = 0; m < K; m++) {
                    for (int n = 0; n < K; n++) {
                        for (int p = 0; p < K; p++) {
                            matC[i][j][m][n] += matA[i][k][m][p] * matB[k][j][p][n];
                        }
                    }
                }
            }
        }
    }
}

__attribute__((target("no-sse,no-avx"), optimize("no-unroll-loops")))
void multiplyMatricesNoVectorization(double matA[][N][K][K], int rowsA, int colsA, double matB[][P][K][K], int rowsB, int colsB, double matC[][P][K][K]) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                for (int m = 0; m < K; m++) {
                    for (int n = 0; n < K; n++) {
                        for (int p = 0; p < K; p++) {
                            matC[i][j][m][n] += matA[i][k][m][p] * matB[k][j][p][n];
                        }
                    }
                }
            }
        }
    }
}

void multiplyMatricesWithManualVectorization(double matA[][N][K][K], int rowsA, int colsA,
                                              double matB[][P][K][K], int rowsB, int colsB,
                                              double matC[][P][K][K]) {

    for (auto i = 0; i < rowsA; i++) {
        for (auto j = 0; j < colsB; j++) {
            for (auto k = 0; k < colsA; k++) {
                for (auto m = 0; m < K; m++) {
                    for (auto n = 0; n < K; n++) {
                        auto rowa = _mm256_load_pd(&matA[i][k][m][0]);
                        auto columna = _mm256_set_pd(matB[k][j][3][n], matB[k][j][2][n], matB[k][j][1][n], matB[k][j][0][n]);
                        auto mula = _mm256_mul_pd(rowa, columna);
                        auto lowa = _mm256_extractf128_pd(mula, 0);
                        auto higha = _mm256_extractf128_pd(mula, 1);
                        auto suma = _mm_add_pd(lowa, higha);               
                        matC[i][j][m][n] += suma[0] + suma[1];
                        }
                    }
                }
            }
        }
    }

double a[M][N][K][K] __attribute__((aligned(32)));
double b[N][P][K][K] __attribute__((aligned(32)));
double c[M][P][K][K] __attribute__((aligned(32)));
double c2[M][P][K][K] __attribute__((aligned(32)));
double c3[M][P][K][K] __attribute__((aligned(32)));

int main() {

    fillFirstMatrix(a, M, N);
    fillSecondMatrix(b, N, P);

    auto start = std::chrono::high_resolution_clock::now();
    multiplyMatrices(a, M, N, b, N, P, c);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Duration with vectorization: " << duration.count() << " seconds" << std::endl;


    start = std::chrono::high_resolution_clock::now();
    multiplyMatricesNoVectorization(a, M, N, b, N, P, c2);
    end = std::chrono::high_resolution_clock::now();

    duration = end - start;
    std::cout << "Duration without vectorization: " << duration.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    multiplyMatricesWithManualVectorization(a, M, N, b, N, P, c3);
    end = std::chrono::high_resolution_clock::now();

    duration = end - start;
    std::cout << "Duration with manual vectorization: " << duration.count() << " seconds" << std::endl;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            for (int m = 0; m < K; m++) {
                for (int n = 0; n < K; n++) {
                    assert(fabs(c[i][j][m][n] - c2[i][j][m][n]) < 0.004);
                }
            }
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            for (int m = 0; m < K; m++) {
                for (int n = 0; n < K; n++) {
                    //std::cout << c[i][j][m][n] << " " << c3[i][j][m][n] << std::endl;
                    assert(fabs(c[i][j][m][n] - c3[i][j][m][n]) < 0.004);
                }
            }
        }
    }

    return 0;
}
