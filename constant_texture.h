#ifndef CONSTANTH
#define CONSTANTH 

#include "texture.h"

class constant_texture : public myTexture {
public:
	__device__ constant_texture() {}

	__device__ constant_texture(vec3 c) : color(c) {}

	__device__ virtual vec3 value(float u, float v, const vec3& p) const {
		return color;
	}

	__device__ virtual vec3 foo(float u, float v, const vec3& p) const {
		return vec3(0, 0, 0);
	}

	vec3 color;
};

#endif