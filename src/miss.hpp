#ifndef MISSH
#define MISSH

#include "common.hpp"
#include "ray.hpp"

class Miss : public Managed {
 public:
  __host__ __device__ virtual float3 sample(const Ray& r) const = 0;
};

class Constant_Background : public Miss {
 public:
  __host__ __device__ Constant_Background(const float3& color) : color(color) {}

  __host__ __device__ virtual float3 sample(const Ray& r) const {
    return color;
  }

  const float3 color;
};

class Gradient_Background : public Miss {
 public:
  __host__ __device__ Gradient_Background(const float3& c0, const float3& c1)
      : c0(c0), c1(c1) {}

  __host__ __device__ virtual float3 sample(const Ray& r) const {
    const float3 unit_direction = normalize(r.direction);
    const float t = 0.5f * (unit_direction.y + 1.f);

    // make gradient color
    return (1.f - t) * c0 + t * c1;
  }

  const float3 c0;
  const float3 c1;
};

#endif