#ifndef HITABLEH
#define HITABLEH

#include "../brdfs/brdfs.hpp"
#include "../common.hpp"
#include "../ray.hpp"

class BRDF;

struct Hit_Record {
  float t;
  float3 hit_point;
  float3 geometric_normal;
  float3 shading_normal;
  BRDF* brdf;
};

class Hitable {
 public:
  __device__ virtual bool hit(const Ray& r, float t_min, float t_max,
                              Hit_Record& rec) const = 0;
};

#endif