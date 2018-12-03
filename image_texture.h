#ifndef IMAGEH
#define IMAGEH 

#include "texture.h"

class image_texture : public myTexture {
public:
	__device__ image_texture() {}

	__device__ image_texture(unsigned char *pixels, int A, int B) : data(pixels), nx(A), ny(B) {}

	__device__ virtual vec3 value(float u, float v, const vec3& p) const {
		int i = u * nx;
		int j = (1 - v) * ny - 0.001;

		if (i < 0)
			i = 0;
		if (i > nx - 1)
			i = nx - 1;

		if (j < 0)
			j = 0;
		if (j > ny - 1)
			j = ny - 1;

		float r = int(data[3 * i + 3 * nx * j]) / 255.0;
		float g = int(data[3 * i + 3 * nx * j + 1]) / 255.0;
		float b = int(data[3 * i + 3 * nx * j + 2]) / 255.0;

		return vec3(r, g, b);
	}

	__device__ virtual vec3 foo(float u, float v, const vec3& p) const {
		return vec3(0, 0, 0);
	}

	unsigned char *data;
	int nx, ny;
};

#endif