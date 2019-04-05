#ifndef BRDFH
#define BRDFH

#include "../common.hpp"
#include "../hitables/hitables.hpp"
#include "../random.hpp"
#include "../ray.hpp"

struct Hit_Record;

// TODO: make three different functions: Surface_Params, Sample and Evaluate

D_FUNCTION float3 random_in_unit_sphere(uint& seed) {
  float z = rnd(seed) * 2.f - 1.f;

  float t = rnd(seed) * 2.f * PI;
  float r = sqrtf((0.f > (1.f - z * z) ? 0.f : (1.f - z * z)));

  float x = r * cosf(t);
  float y = r * sinf(t);

  float3 res = make_float3(x, y, z);
  res *= powf(rnd(seed), 1.f / 3.f);

  return res;
}

// Returns true if a refraction was possible
D_FUNCTION bool refract(const float3& v,         // incident vector
                        const float3& n,         // normal vector
                        const float ni_over_nt,  // ratio of refraction
                        float3& refracted)       // resultant vector
{
  float3 uv = normalize(v);  // normalize incident vector

  // Snell's Law
  float dt = dot(uv, n);
  float discriminant = 1.f - ni_over_nt * ni_over_nt * (1.f - dt * dt);

  if (discriminant > 0) {
    refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
    return true;
  } else
    return false;
}

// Christophe Schlick's reflectivity approximation
D_FUNCTION float schlick(float cosine, float ref_idx) {
  float r0 = (1.f - ref_idx) / (1.f + ref_idx);
  r0 = r0 * r0;

  return r0 + (1.f - r0) * powf(1.f - cosine, 5.f);
}

class BRDF : public Managed {
 public:
  __host__ __device__ virtual bool scatter(const Ray& r_in,
                                           const Hit_Record& rec,
                                           float3& attenuation, Ray& scattered,
                                           uint& seed) const = 0;
};

#endif