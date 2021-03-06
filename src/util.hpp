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

// swaps values of T class objects a and b
template <class T>
inline __host__ __device__ void swap_values(T &a, T &b) {
  T c(a);
  a = b;
  b = c;
}

// Removes nan values from vector
inline __device__ float3 de_nan(const float3 &c) {
  float3 temp = c;

  if (!(temp.x == temp.x)) temp.x = 0.f;
  if (!(temp.y == temp.y)) temp.y = 0.f;
  if (!(temp.z == temp.z)) temp.z = 0.f;

  return temp;
}

inline __device__ float get_component(const float3 &c, const int i) {
  switch (i) {
    case 0:
      return c.x;
    case 1:
      return c.y;
    default:
      return c.z;
  }
}

#endif