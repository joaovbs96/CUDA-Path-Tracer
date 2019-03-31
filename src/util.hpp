#ifndef UTILH
#define UTILH

#include "common.hpp"
#include "cutil_math.h"

// Returns vector with square rooted coordinates, one by one
inline __host__ __device__ float3 sqrtf(const float3 &v) {
  return make_float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
}

// Returns vector with squared coordinates, one by one
inline __host__ __device__ float3 sqrf(const float3 &v) {
  return make_float3(v.x * v.x, v.y * v.y, v.z * v.z);
}

// Removes nan values from vector
inline __device__ float3 de_nan(const float3 &c) {
  float3 temp = c;

  if (!(temp.x == temp.x)) temp.x = 0.f;
  if (!(temp.y == temp.y)) temp.y = 0.f;
  if (!(temp.z == temp.z)) temp.z = 0.f;

  return temp;
}

#endif