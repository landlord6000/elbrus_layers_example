#pragma once

#include <memory> 
#include <cstring>

#include <e2kbuiltin.h>
#include <e2kintrin.h>

#include "matrix.h"

typedef __attribute((vector_size(16))) long long v2uint64_t;
typedef __attribute((vector_size(16))) long long* v2uint64_t_ptr;

void pool_avg_global(float *__restrict__ in, float *__restrict__ out, float *__restrict__  XY, long B, long X, long Y, long L);
