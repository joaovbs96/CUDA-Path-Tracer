#ifndef LAMBERTIANH
#define LAMBERTIANH

#include "brdfs.hpp"

class Lambertian : public BRDF {
 public:
  __device__ Lambertian(const float3& a) : albedo(a) {}

  __device__ virtual bool scatter(const Ray& r_in, const Hit_Record& rec,
                                  float3& attenuation, Ray& scattered,
                                  uint& seed) const {
    float3 target = rec.geometric_normal + random_in_unit_sphere(seed);
    scattered = Ray(rec.hit_point, target);

    attenuation = albedo;

    return true;
  }

  const float3 albedo;
};

#endif