#ifndef LAMBERTIANH
#define LAMBERTIANH

#include "material.h"
#include "texture.h"

class lambertian : public material {
public:
	__device__ lambertian(myTexture *a) : albedo(a) {}

	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const {
		vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
		scattered = ray(rec.p, target - rec.p, r_in.time());
		attenuation = albedo->value(0, 0, rec.p);
		return true;
	}

	myTexture *albedo;
};

#endif // !LAMBERTIANH