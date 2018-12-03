#ifndef METALH
#define METALH

#include "material.h"

class metal : public material {
public:
	__device__ metal(myTexture *a, float f) : albedo(a) {
		if (f < 1) 
			fuzz = f; 
		else 
			fuzz = 1; 
	}

	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const {
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state), r_in.time());
		attenuation = albedo->value(0, 0, rec.p);
		
		return (dot(scattered.direction(), rec.normal) > 0.0f);
	}
	
	myTexture *albedo;
	float fuzz;
};

#endif // !METALH