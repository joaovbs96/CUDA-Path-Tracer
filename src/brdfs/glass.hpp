#ifndef GLASSH
#define GLASSH

#include "brdfs.hpp"

class Glass : public BRDF {
 public:
  __device__ Glass(const float3& a, float ref_idx)
      : albedo(a), ref_idx(ref_idx) {}

  __device__ virtual bool scatter(const Ray& r_in, const Hit_Record& rec,
                                  float3& attenuation, Ray& scattered,
                                  uint& seed) const {
    float3 outward_normal;
    float3 reflected = reflect(r_in.direction, rec.geometric_normal);
    float ni_over_nt;

    // Colored Glass
    attenuation = albedo;

    float3 refracted;
    float reflect_prob;
    float cosine;

    if (dot(r_in.direction, rec.geometric_normal) > 0.0f) {
      outward_normal = -1.f * rec.geometric_normal;
      ni_over_nt = ref_idx;
      cosine =
          dot(r_in.direction, rec.geometric_normal) / length(r_in.direction);
      cosine = sqrtf(1.f - ref_idx * ref_idx * (1 - cosine * cosine));
    } else {
      outward_normal = rec.geometric_normal;
      ni_over_nt = 1.f / ref_idx;
      cosine =
          -dot(r_in.direction, rec.geometric_normal) / length(r_in.direction);
    }

    if (refract(r_in.direction, outward_normal, ni_over_nt, refracted))
      reflect_prob = schlick(cosine, ref_idx);
    else
      reflect_prob = 1.0f;

    if (rnd(seed) < reflect_prob)
      scattered = Ray(rec.hit_point, reflected);
    else
      scattered = Ray(rec.hit_point, refracted);

    return true;
  }

  const float3 albedo;
  const float ref_idx;
};

#endif