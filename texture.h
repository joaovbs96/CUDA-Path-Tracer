#ifndef TEXTUREH
#define TEXTUREH 

#include "vec3.h"

class myTexture {
public:
	__device__ virtual vec3 foo(float u, float v, const vec3& p) const = 0;
	__device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

#endif