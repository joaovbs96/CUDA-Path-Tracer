#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere : public hitable {
public:
	__device__ sphere() { }
	
	__device__ sphere(vec3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m) { };

	__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;

	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;

	__device__ virtual hitableType getType() const {
		return SPHERE;
	}

	vec3 center;
	float radius;
	material *mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	vec3 oc = r.origin() - center;

	// if the ray hits the sphere, the following equation has two roots:
	// tdot(B, B) + 2tdot(B,A-C) + dot(A-C,A-C) - R = 0

	// Using Bhaskara's Formula, we have:
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;

	// if the discriminant is higher than 0, there's a hit.
	if (discriminant > 0) {
		// first root of the sphere equation:
		float temp = (-b - sqrt(discriminant)) / a;

		// surface normals: vector perpendicular to the surface,
		// points out of it by convention.

		// for a sphere, its normal is in the (hitpoint - center).

		// if the first root was a hit,
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			get_sphere_uv((rec.p - center) / radius, rec.u, rec.v);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}

		// if the second root was a hit,
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			get_sphere_uv((rec.p - center) / radius, rec.u, rec.v);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
	}

	// otherwise, there's no hit.
	return false;
}

__device__ bool sphere::bounding_box(float t0, float t1, aabb& box) const {
	box = aabb(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius));

	return true;
}


#endif