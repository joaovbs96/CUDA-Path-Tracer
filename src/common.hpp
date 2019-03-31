#pragma once

// Common defines and includes

#include "cutil_math.h"

// To make use in Device only functions
#define D_FUNCTION __forceinline__ __device__

// To make use in Global functions/kernels
#define G_FUNCTION __global__

// To make use in Device and Host functions
#define DH_FUNCTION __forceinline__ __device__ __host__

// Math defines
#ifndef PI
#define PI 3.141592654f
#endif

// Axis type
typedef enum { X_AXIS, Y_AXIS, Z_AXIS } AXIS;