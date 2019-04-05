#pragma once

// Common defines and includes

#include "cutil_math.h"

// To make use in Device only functions
#define D_FUNCTION __forceinline__ __device__ __host__

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

// Managed Base Class -- inherit from this to automatically
// allocate objects in Unified Memory
class Managed {
 public:
  __host__ __device__ void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }

  __host__ __device__ void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};