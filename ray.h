#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray {
public:
	__device__ ray() {}

	// A: ray origin
	// B: ray direction

	__device__ ray(const vec3& a, const vec3& b, float ti = 0.0) {
		A = a;
		B = b;
		_time = ti;
	}

	__device__ vec3 origin() const {
		return A;
	}

	__device__ vec3 direction() const {
		return B;
	}

	// t: parameter that changes 'destination' along ray/half-line
	__device__ vec3 point_at_parameter(float t) const {
		return A + (t*B);
	}

	// used in motion blur
	// motion blur is an average of what's seen by the camera shutter 
	// during the time it stays open
	__device__ float time() const {
		return _time;
	}

	vec3 A;
	vec3 B;
	float _time;
};

#endif