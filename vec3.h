#ifndef VEC3H
#define VEC3H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <float.h>
#include <random>

#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif

__host__ __device__ inline float clamp(float value) {
	return value > 1.0f ? 1.0f : value;
}

#define RNG() curand_uniform(&local_rand_state)

class vec3 {
public:
	__host__ __device__ vec3() {}

	__host__ __device__ vec3(float e0, float e1, float e2) {
		e[0] = e0;
		e[1] = e1;
		e[2] = e2;
		uv[0] = 0.0;
		uv[1] = 0.0;
	}

	// optional parameters UV for shading
	__host__ __device__ vec3(float e0, float e1, float e2, float uu, float vv) {
		e[0] = e0;
		e[1] = e1;
		e[2] = e2;
		uv[0] = uu;
		uv[1] = vv;
	}

	// Basic return functions
	__host__ __device__ inline float x() const {
		return e[0];
	}

	__host__ __device__ inline float y() const {
		return e[1];
	}

	__host__ __device__ inline float z() const {
		return e[2];
	}

	__host__ __device__ inline float r() const {
		return e[0];
	}

	__host__ __device__ inline float g() const {
		return e[1];
	}

	__host__ __device__ inline float b() const {
		return e[2];
	}

	__host__ __device__ inline float u() const {
		return uv[0];
	}

	__host__ __device__ inline float v() const {
		return uv[1];
	}

	// Operator overloading
	// +1 * vec3
	__host__ __device__ inline const vec3& operator+() const {
		return *this;
	}

	// -1 * vec3
	__host__ __device__ inline vec3 operator-() const {
		return vec3(-e[0], -e[1], -e[2]);
	}

	// vec3[i]
	__host__ __device__ inline float operator[](int i) const {
		return e[i];
	}

	// reference to vec3[i]
	__host__ __device__ inline float& operator[](int i) {
		return e[i];
	}

	// overloading of +=, -=, etc
	__host__ __device__ inline vec3& operator+=(const vec3 &v2);
	__host__ __device__ inline vec3& operator-=(const vec3 &v2);
	__host__ __device__ inline vec3& operator*=(const vec3 &v2);
	__host__ __device__ inline vec3& operator/=(const vec3 &v2);
	__host__ __device__ inline vec3& operator*=(const float t);
	__host__ __device__ inline vec3& operator/=(const float t);

	// euclidean distance of vector
	__host__ __device__ inline float length() const {
		return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
	}
	// square of distance
	__host__ __device__ inline float squared_length() const {
		return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
	}

	__host__ __device__ inline void make_unit_vector();

	float e[3];
	float uv[2];
};

inline std::istream& operator>>(std::istream &is, vec3 &t) {
	is >> t.e[0] >> t.e[1] >> t.e[2];
	return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
	os << t.e[0] << " " << t.e[1] << " " << t.e[2];
	return os;
}

__host__ __device__ inline void vec3::make_unit_vector() {
	float k = 1.0f / (sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]));
	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
}

__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
	return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
	return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
	return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
	return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
	return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v, float t) {
	return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
	return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2) {
	return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2) {
	return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
		-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0]),
		(v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3 &v) {
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];
	return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3 &v) {
	e[0] -= v.e[0];
	e[1] -= v.e[1];
	e[2] -= v.e[2];
	return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3 &v) {
	e[0] *= v.e[0];
	e[1] *= v.e[1];
	e[2] *= v.e[2];
	return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3 &v) {
	e[0] /= v.e[0];
	e[1] /= v.e[1];
	e[2] /= v.e[2];
	return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t) {
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t) {
	float k = 1.0f / t;

	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
	return *this;
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
	return v / v.length();
}

__device__ inline vec3 random_cosine_direction(curandState *local_rand_state) {
	float r1 = curand_uniform(local_rand_state);
	float r2 = curand_uniform(local_rand_state);
	
	float phi = 2 * M_PI * r1;
	
	float x = cos(phi) * 2 * sqrt(r2);
	float y = sin(phi) * 2 * sqrt(r2);
	float z = sqrt(1 - r2);

	return vec3(x, y, z);
}

__device__ inline vec3 random_to_sphere(float radius, float distance_squared, curandState *local_rand_state) {
	float r1 = curand_uniform(local_rand_state);
	float r2 = curand_uniform(local_rand_state);

	float phi = 2 * M_PI * r1;

	float z = 1 + r2 * (sqrt(1 - radius * radius / distance_squared) - 1);
	float x = cos(phi) * sqrt(1 - z * z);
	float y = sin(phi) * sqrt(1 - z * z);

	return vec3(x, y, z);
}

__host__ __device__ inline vec3 de_nan(const vec3& c) {
	vec3 temp = c;
	if (!(temp[0] == temp[0])) temp[0] = 0;
	if (!(temp[1] == temp[1])) temp[1] = 0;
	if (!(temp[2] == temp[2])) temp[2] = 0;
	return temp;
}

#endif