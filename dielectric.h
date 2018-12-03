#ifndef DIELECTRICH
#define DIELECTRICH

#include "material.h"

class dielectric : public material {
public:
	__device__ dielectric(float ri) : ref_idx(ri) {}

	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const {
		vec3 outward_normal;
		vec3 reflected = reflect(r_in.direction(), rec.normal);
		float ni_over_nt;
		attenuation = vec3(1.0, 1.0, 1.0);
		vec3 refracted;
		float reflect_prob;
		float cosine;

		if (dot(r_in.direction(), rec.normal) > 0.0f) {
			outward_normal = -rec.normal;
			ni_over_nt = ref_idx;
			cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
			cosine = sqrt(1.0f - ref_idx * ref_idx*(1 - cosine * cosine));
		}
		else {
			outward_normal = rec.normal;
			ni_over_nt = 1.0f / ref_idx;
			cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
		}

		if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
			reflect_prob = schlick(cosine, ref_idx);
		else
			reflect_prob = 1.0f;

		if (curand_uniform(local_rand_state) < reflect_prob)
			scattered = ray(rec.p, reflected);
		else
			scattered = ray(rec.p, refracted);

		return true;
	}

	float ref_idx;
};
#endif // !DIELECTRICH
