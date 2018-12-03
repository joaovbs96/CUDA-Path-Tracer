#ifndef AABBH
#define AABBH

#include "vec3.h"
#include "ray.h"

// if (a < b), return a, else return b
__device__ inline float ffmin(float a, float b) {
	return a < b ? a : b;
}

// if (a > b), return a, else return b
__device__ inline float ffmax(float a, float b) {
	return a > b ? a : b;
}

// return pairwise min vector
__device__ vec3 minVec(vec3 a, vec3 b) {
	return vec3(ffmin(a.x(), b.x()), ffmin(a.y(), b.y()), ffmin(a.z(), b.z()));
}

// return pairwise max vector
__device__ vec3 maxVec(vec3 a, vec3 b) {
	return vec3(ffmax(a.x(), b.x()), ffmax(a.y(), b.y()), ffmax(a.z(), b.z()));
}

// return max component of vector
__device__ float max_component(vec3 a) {
	return ffmax(ffmax(a.x(), a.y()), a.z());
}

// return max component of vector
__device__ float min_component(vec3 a) {
	return ffmin(ffmin(a.x(), a.y()), a.z());
}

class aabb {
public:
	__device__ aabb() {}

	__device__ aabb(const vec3& a, const vec3& b) {
		_min = a;
		_max = b;
	}

	__device__ vec3 min() const {
		return _min;
	}

	__device__ vec3 max() const {
		return _max;
	}

	__device__ bool hit(const ray& r, float tmin, float tmax) const {
		vec3 invRayDir(1.0f / r.direction().x(), 1.0f / r.direction().y(), 1.0f / r.direction().z());
		
		vec3 t0 = (_min - r.origin()) * invRayDir;
		vec3 t1 = (_max - r.origin()) * invRayDir;
		vec3 t_min = minVec(t0, t1), t_max = maxVec(t0, t1);
		bool temp = max_component(t_min) <= min_component(t_max);
		
		return temp;
	}

	vec3 _min, _max;
};

// computes bounding box of two boxes
__device__ aabb surrounding_box(aabb box0, aabb box1) {
	vec3 small(fmin(box0.min().x(), box1.min().x()),
		fmin(box0.min().y(), box1.min().y()),
		fmin(box0.min().z(), box1.min().z()));

	vec3 big(fmax(box0.max().x(), box1.max().x()),
		fmax(box0.max().y(), box1.max().y()),
		fmax(box0.max().z(), box1.max().z()));

	return aabb(small, big);
}

#endif // !AABBH
