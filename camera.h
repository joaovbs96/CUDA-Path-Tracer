#ifndef CAMERAH
#define CAMERAH

#include "ray.h"

__device__ vec3 random_in_unit_disk(curandState *local_rand_state) {
	float a = curand_uniform(local_rand_state) * 2.0f * 3.1415926f;
	vec3 xy(sin(a), cos(a), 0);
	xy *= sqrt(curand_uniform(local_rand_state));
	return xy;
}

class camera {
public:
	__device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist, float t0, float t1) {
		lens_radius = aperture / 2;

		// shutter is open between t0 and t1
		time0 = t0;
		time1 = t1;
		
		float theta = vfov * M_PI / 180;
		float half_height = tan(theta / 2);
		float half_width = aspect * half_height;
		
		// where camera is looking from
		origin = lookfrom;

		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);

		lower_left_corner = origin - focus_dist * (half_width * u + half_height * v + w);
		horizontal = 2 * half_width * focus_dist * u;
		vertical = 2 * half_height * focus_dist * v;
	}

	// trace a new ray
	__device__ ray get_ray(float s, float t, curandState *local_rand_state) {
		vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
		vec3 offset = u * rd.x() + v * rd.y();
		float time = time0 + curand_uniform(local_rand_state) * (time1 - time0);
		return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset, time);
	}

	vec3 origin;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	float time0, time1;
	float lens_radius;
};

#endif