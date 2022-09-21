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

#include "defines.h"

void conv_gradient_general(float *__restrict__ in, float *__restrict__ prm, float *__restrict__ out, float *__restrict__ bs, \
    int B, int X, int Y, int Xout, int Yout, int L, int F, int R, int S, int P);