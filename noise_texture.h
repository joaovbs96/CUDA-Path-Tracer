#ifndef NOISE_TEXTUREH
#define NOISE_TEXTUREH

#include "texture.h"
#include "perlin.h"

class noise_texture : public myTexture {
public:
	__device__ noise_texture() {}

	__device__ noise_texture(float sc, curandState *local_rand_state) : scale(sc) {
		noise = perlin(local_rand_state);
	}

	__device__ virtual vec3 value(float u, float v, const vec3& p) const {
		return vec3(1, 1, 1) * 0.5 * (1 + sin(scale * p.z() + 10 * noise.turb(p)));
	}

	__device__ virtual vec3 foo(float u, float v, const vec3& p) const {
		return vec3(0, 0, 0);
	}

	perlin noise;
	float scale;
};


#endif // !NOISE_TEXTUREH
