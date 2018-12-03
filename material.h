#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include "ray.h"
#include "hitable.h"

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
	float z = curand_uniform(local_rand_state) * 2.0f - 1.0f;
	float t = curand_uniform(local_rand_state) * 2.0f * 3.1415926f;
	float r = sqrt((0.0f > (1.0f - z * z) ? 0.0f : (1.0f - z * z)));
	float x = r * cos(t);
	float y = r * sin(t);
	vec3 res(x, y, z);
	res *= powf(curand_uniform(local_rand_state), 1.0f / 3.0f);
	return res;
}

__device__ float schlick(float cosine, float ref_idx) {
	float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
	vec3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float discriminant = 1.0f - ni_over_nt * ni_over_nt*(1 - dt * dt);
	if (discriminant > 0) {
		refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
		return true;
	}
	else
		return false;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2.0f * dot(v, n) * n;
}

class material {
public:
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const = 0;
};

#endif