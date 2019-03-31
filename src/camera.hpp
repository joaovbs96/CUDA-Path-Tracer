#ifndef CAMERAH
#define CAMERAH

#include "common.hpp"
#include "random.hpp"
#include "ray.hpp"

D_FUNCTION float3 random_in_unit_disk(uint &seed) {
  float a = rnd(seed) * 2.f * PI;

  float3 xy = make_float3(sin(a), cos(a), 0);
  xy *= sqrt(rnd(seed));

  return xy;
}

class Camera {
 public:
  __device__ Camera(float3 lookfrom, float3 lookat, float3 vup, float vfov,
                    float aspect, float aperture, float focus_dist) {
    lens_radius = aperture / 2;

    float theta = vfov * PI / 180.f;
    float half_height = tanf(theta / 2.f);
    float half_width = aspect * half_height;

    origin = lookfrom;

    w = normalize(lookfrom - lookat);
    u = normalize(cross(vup, w));
    v = cross(w, u);

    lower_left_corner = origin;
    lower_left_corner -= focus_dist * (half_width * u + half_height * v + w);

    horizontal = 2.f * half_width * focus_dist * u;
    vertical = 2.f * half_height * focus_dist * v;
  }

  __device__ Ray get_ray(float s, float t, uint &seed) {
    float3 rd = lens_radius * random_in_unit_disk(seed);
    float3 offset = u * rd.x + v * rd.y;
    float3 direction =
        lower_left_corner + s * horizontal + t * vertical - origin - offset;

    return Ray(origin + offset,  // ray origin
               direction);       // ray direction
  }

  float3 origin;
  float3 lower_left_corner;
  float3 horizontal;
  float3 vertical;
  float3 u, v, w;
  float lens_radius;
};

#endif