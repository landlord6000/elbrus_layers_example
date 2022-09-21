#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <memory> 

#include <e2kbuiltin.h>
#include <e2kintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void conv_backward_general(float *__restrict__ in, float *__restrict__ prm, float *__restrict__ prm_new, float *__restrict__ out, \
    int B, int X, int Y, int Xout, int Yout, int L, int F, int Rx, int Ry, int S, int Px, int Py);