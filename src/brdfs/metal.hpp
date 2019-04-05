#ifndef METALH
#define METALH

#include "brdfs.hpp"

class Metal : public BRDF {
 public:
  __host__ __device__ Metal(const float3& a, const float f)
      : albedo(a), roughness(clamp(f, 0.f, 1.f)) {}

  __host__ __device__ virtual bool scatter(const Ray& r_in,
                                           const Hit_Record& rec,
                                           float3& attenuation, Ray& scattered,
                                           uint& seed) const {
    float3 reflected = reflect(normalize(r_in.direction), rec.geometric_normal);
    float3 direction = reflected + roughness * random_in_unit_sphere(seed);
    scattered = Ray(rec.hit_point, direction);

    attenuation = albedo;

    return (dot(scattered.direction, rec.geometric_normal) > 0.0f);
  }

  const float3 albedo;
  const float roughness;
};

#endif