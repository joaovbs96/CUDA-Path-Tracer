#ifndef CHECKERH
#define CHECKERH 

#include "texture.h"

class checker_texture : public myTexture {
public:
	__device__ checker_texture() {}

	__device__ checker_texture(myTexture *t0, myTexture *t1) : even(t0), odd(t1) {}

	__device__ virtual vec3 value(float u, float v, const vec3& p) const {
		float sines = sin(10 * p.x()) * sin(10 - p.y()) * sin(10 * p.z());

		if (sines < 0)
			return odd->value(u, v, p);
		else
			return even->value(u, v, p);
	}

	__device__ virtual vec3 foo(float u, float v, const vec3& p) const {
		return vec3(0,0,0);
	}

	myTexture *odd;
	myTexture *even;
};

#endif