#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand_kernel.h>

#include <iostream>
#include <string>
#include <time.h>

#include "hitable_list.h"
#include "sphere.h"
#include "moving_sphere.h"
#include "bvh_node.h"

#include "lambertian.h"
#include "metal.h"
#include "dielectric.h"

#include "checker_texture.h"
#include "image_texture.h"
#include "constant_texture.h"
#include "noise_texture.h"

#include "ray.h"
#include "camera.h"

#pragma warning(push, 0)        
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma warning(pop)

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__device__ vec3 color(const ray& r, hitable **world, int depth, curandState *local_rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);

	for (int i = 0; i < 50; i++) {
		hit_record rec;

		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;

			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else {
				return vec3(0.0, 0.0, 0.0);
			}
		}
		else {
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void render(vec3 *arr, int width, int height, camera **cam, hitable **world, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= width) || (j >= height))
		return;

	// get local random state
	int pixel_index = j * width + i;
	curandState local_rand_state = rand_state[pixel_index];

	float u = float(i + curand_uniform(&local_rand_state)) / float(width);
	float v = float(j + curand_uniform(&local_rand_state)) / float(height);

	ray r = (*cam)->get_ray(u, v, &local_rand_state);

	arr[pixel_index] = color(r, world, 0, &local_rand_state);

	rand_state[pixel_index] = local_rand_state;
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= max_x) || (j >= max_y))
		return;
	
	int pixel_index = j * max_x + i;
	
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void rand_init(curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, rand_state);
	}
}


__global__ void random_moving(int width, int height, hitable **d_list, hitable **d_world, camera **d_camera, curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;
		
		myTexture *checker = new checker_texture(new constant_texture(vec3(0.2, 0.3, 0.1)), new constant_texture(vec3(0.9, 0.9, 0.9)));
		d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000, new lambertian(checker));
				
		int i = 1;
		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				float choose_mat = RNG();
				vec3 center(a + RNG(), 0.2, b + RNG());
				if (choose_mat < 0.8f) {
					d_list[i++] = new moving_sphere(center, center + vec3(0, 0.5 * RNG(), 0), 0.0, 1.0, 0.2, new lambertian(new constant_texture(vec3(RNG()*RNG(), RNG()*RNG(), RNG()*RNG()))));
				}
				else if (choose_mat < 0.95f) {
					d_list[i++] = new sphere(center, 0.2, new metal(new constant_texture(vec3(0.5f*(1.0f + RNG()), 0.5f*(1.0f + RNG()), 0.5f*(1.0f + RNG()))), 0.5f*RNG()));
				}
				else {
					d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
		d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
		d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(new constant_texture(vec3(0.4, 0.2, 0.1))));
		d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(new constant_texture(vec3(0.7, 0.6, 0.5)), 0.0));
		*rand_state = local_rand_state;
		//*d_world = new bvh_node(d_list, 22 * 22 + 1 + 3, 0.0, 1.0, &local_rand_state);
		*d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);

		vec3 lookfrom(13, 2, 3);
		vec3 lookat(0, 0, 0);
		float dist_to_focus = 10.0; (lookfrom - lookat).length();
		float aperture = 0.1;
		*d_camera = new camera(lookfrom, lookat, vec3(0, 1, 0), 30.0, float(width) / float(height), aperture, dist_to_focus, 0.0, 1.0);
	}
}

__global__ void two_perlin_spheres(int width, int height, hitable **d_list, hitable **d_world, camera **d_camera, curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;

		material *noisy = new lambertian(new noise_texture(4.0, &local_rand_state));
		d_list[0] = new sphere(vec3(0, -1000, 0), 1000, noisy);

		noisy = new lambertian(new noise_texture(4.0, &local_rand_state));
		d_list[1] = new sphere(vec3(0, 2, 0), 2, noisy);

		*rand_state = local_rand_state;
		//*d_world = new bvh_node(d_list, 22 * 22 + 1 + 3, 0.0, 1.0, &local_rand_state);
		*d_world = new hitable_list(d_list, 2);

		vec3 lookfrom(13, 2, 3);
		vec3 lookat(0, 0, 0);
		float dist_to_focus = 10.0; (lookfrom - lookat).length();
		float aperture = 0.1;
		*d_camera = new camera(lookfrom, lookat, vec3(0, 1, 0), 30.0, float(width) / float(height), aperture, dist_to_focus, 0.0, 1.0);
	}
}

__global__ void random_scene(int width, int height, unsigned char *img, int nx, int ny, hitable **d_list, hitable **d_world, camera **d_camera, curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;
		d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000, new lambertian(new constant_texture(vec3(0.5, 0.5, 0.5))));
		int i = 1;
		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				float choose_mat = RNG();
				vec3 center(a + RNG(), 0.2, b + RNG());
				if (choose_mat < 0.8f) {
					d_list[i++] = new sphere(center, 0.2, new lambertian(new constant_texture(vec3(RNG()*RNG(), RNG()*RNG(), RNG()*RNG()))));
				}
				else if (choose_mat < 0.95f) {
					d_list[i++] = new sphere(center, 0.2,
						new metal(new constant_texture(vec3(0.5f*(1.0f + RNG()), 0.5f*(1.0f + RNG()), 0.5f*(1.0f + RNG()))), 0.5f*RNG()));
				}
				else {
					d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
		d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
		d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(new constant_texture(vec3(0.4, 0.2, 0.1))));
		d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new lambertian(new image_texture(img, nx, ny)));
		//d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(new constant_texture(vec3(0.7, 0.6, 0.5)), 0.0));
		*rand_state = local_rand_state;
		//*d_world = new bvh_node(d_list, 22 * 22 + 1 + 3, 0.0, 1.0, &local_rand_state);
		*d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);

		vec3 lookfrom(13, 2, 3);
		vec3 lookat(0, 0, 0);
		float dist_to_focus = 10.0; (lookfrom - lookat).length();
		float aperture = 0.1;
		*d_camera = new camera(lookfrom, lookat, vec3(0, 1, 0), 30.0, float(width) / float(height), aperture, dist_to_focus, 0.0, 1.0);
	}
}

__device__ void deleteList(hitable **d_list) {
	for (int i = 0; i < ((hitable_list *)d_list)->list_size - 1; i++) {
		switch (d_list[i]->getType()) {
		case SPHERE:
			printf("sphere\n");
			delete ((sphere *)d_list[i])->mat_ptr;
			delete d_list[i];
			break;
		case MOVING_SPHERE:
			printf("moving_sphere\n");
			delete ((moving_sphere *)d_list[i])->mat_ptr;
			delete d_list[i];
			break;
		case LIST:
			printf("list\n");
			//deletes each one of the list's members
			deleteList((hitable **)d_list[i]);
			break;
		default:
			printf("error\n");
		}
	}
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
	deleteList((hitable**)d_list);
	delete *d_world;
	delete *d_camera;
}


int main() {
	int samples = 100;
	int width = 1200;
	int height = 800;
	int tx = 8;
	int ty = 8;
	int scene = 0;

	std::cerr << "Rendering a " << width << "x" << height << " image, with " << samples << " samples ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	// allocate pixel arrays
	unsigned char *arr = (unsigned char *)malloc(width * height * 3 * sizeof(unsigned char));
	vec3 *accArr = (vec3 *)malloc(width * height * sizeof(vec3));
	for (int j = height - 1; j >= 0; j--)
		for (int i = 0; i < width; i++) {
			int pixel_index = j * width + i;
			accArr[pixel_index] = vec3(0, 0, 0);
		}
	
	vec3 *tempArr;
	tempArr = (vec3 *)malloc(width * height * sizeof(vec3));
	checkCudaErrors(cudaMallocManaged((void **)&tempArr, width * height * sizeof(vec3)));

	// allocate random state array
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, width * height * sizeof(curandState)));
	curandState *d_rand_state2;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

	// we need that 2nd random state to be initialized for the world creation
	rand_init<<<1, 1 >>>(d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// allocate world
	hitable **d_world;
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));

	// allocate camera
	camera **d_camera;
	checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

	int nx, ny, nn;
	unsigned char *tex_data = stbi_load("grr.jpg", &nx, &ny, &nn, 0);

	hitable **d_list;
	int num_hitables;
	std::string filename;
	switch (scene) {
		case 0: // In One Weekend Random Scene
			filename = "InOneWeekend_Ch2_";

			// allocate list of hitables
			num_hitables = 22 * 22 + 1 + 3;
			checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hitable *)));

			// create world/scene
			random_scene<<<1, 1>>>(width, height, tex_data, nx, ny, d_list, d_world, d_camera, d_rand_state2);
			break;

		case 1:// The Next Week Random Moving Scene
			filename = "NextWeek_Ch3_";

			// allocate list of hitables
			num_hitables = 22 * 22 + 1 + 3;
			checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hitable *)));

			// create world/scene
			random_moving<<<1, 1>>>(width, height, d_list, d_world, d_camera, d_rand_state2);
		
		default:// The Next Week Perlin Spheres
			filename = "NextWeek_Ch4_";

			// allocate list of hitables
			num_hitables = 2;
			checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hitable *)));

			// create world/scene
			two_perlin_spheres<<<1, 1>>>(width, height, d_list, d_world, d_camera, d_rand_state2);

	}
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// init time
	clock_t start, stop;
	start = clock();

	// Render
	dim3 blocks(width / tx + 1, height / ty + 1);
	dim3 threads(tx, ty);

	printf("Initiating Render\n");
	render_init<<<blocks, threads>>>(width, height, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// print init time
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds to init render.\n";
	start = clock();

	for (int s = 0; s < samples; s++) {
		printf("Progress: %.2f%%     \r", (s * 100.0f/samples));
		render <<<blocks, threads>>>(tempArr, width, height, d_camera, d_world, d_rand_state);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		// accumulate samples
		for (int j = height - 1; j >= 0; j--)
			for (int i = 0; i < width; i++) {
				int pixel_index = j * width + i;
				accArr[pixel_index] += tempArr[pixel_index];
			}

	}
	printf("\n\n");

	// average samples
	for (int j = height - 1; j >= 0; j--)
		for (int i = 0; i < width; i++) {
			vec3 col = accArr[j * width + i];
			
			col /= float(samples); // average samples
			col = vec3(sqrt(col.r()), sqrt(col.g()), sqrt(col.b())); // gamma correction

			int pixel_index = (height - j - 1) * 3 * width + 3 * i;
			arr[pixel_index + 0] = int(255.99 * clamp(col.r()));
			arr[pixel_index + 1] = int(255.99 * clamp(col.g()));
			arr[pixel_index + 2] = int(255.99 * clamp(col.b()));
		}
	
	// print render time
	stop = clock();
	timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds to render.\n";

	// save image
	std::string output = "output/";
	output.append(filename);
	output.append(std::to_string(width));
	output.append("x");
	output.append(std::to_string(height));
	output.append("_");
	output.append(std::to_string(samples));
	output.append(".png");
	stbi_write_png((char*)output.c_str(), width, height, 3, arr, 0);
		
	// clean up
	checkCudaErrors(cudaDeviceSynchronize());

	// scene functions
	// TODO: not freeing the scene correctly
	//free_world<<<1, 1>>>(d_list, d_world, d_camera);	
	// /scene functions

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(tempArr));

	// useful for cuda-memcheck --leak-check full
	cudaDeviceReset();
}