#ifndef SPHEREH
#define SPHEREH

#include "../common.hpp"
#include "hitables.hpp"

class Sphere : public Hitable {
 public:
  __device__ Sphere() {}

  __device__ Sphere(float3 center, float radius, BRDF* brdf)
      : center(center), radius(radius), brdf(brdf){};

  __device__ virtual bool hit(const Ray& r, float tmin, float tmax,
                              Hit_Record& rec) const;

  __device__ virtual void free() const { delete brdf; }

  float3 center;
  float radius;
  BRDF* brdf;
};

// Sphere Intersection Function
__device__ bool Sphere::hit(const Ray& ray, float t_min, float t_max,
                            Hit_Record& rec) const {
  float3 oc = ray.origin - center;

  float a = dot(ray.direction, ray.direction);
  float b = dot(oc, ray.direction);
  float c = dot(oc, oc) - radius * radius;

  float discriminant = b * b - a * c;

  // Check both roots of the equation
  if (discriminant > 0) {
    // First Root
    float temp = (-b - sqrt(discriminant)) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.hit_point = ray.point_at_parameter(rec.t);
      rec.geometric_normal = (rec.hit_point - center) / radius;
      rec.brdf = brdf;
      return true;
    }

    // Second Root
    temp = (-b + sqrt(discriminant)) / a;
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.hit_point = ray.point_at_parameter(rec.t);
      rec.geometric_normal = (rec.hit_point - center) / radius;
      rec.brdf = brdf;
      return true;
    }
  }

  // We had no hits, return false
  return false;
}

#endif