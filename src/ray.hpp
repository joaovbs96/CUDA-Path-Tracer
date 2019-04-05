#ifndef RAYH
#define RAYH

#include "common.hpp"
#include "cutil_math.h"

class Ray {
 public:
  __host__ __device__ Ray() {}

  __host__ __device__ Ray(const float3& origin, const float3& direction)
      : origin(origin), direction(direction) {}

  __host__ __device__ float3 point_at_parameter(float t) const {
    return origin + t * direction;
  }

  float3 origin;
  float3 direction;
};

#endif