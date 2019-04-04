#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cutil_math.h"

#include <time.h>
#include <iostream>
#include <string>

#include "camera.hpp"
#include "common.hpp"
#include "miss.hpp"
#include "random.hpp"
#include "util.hpp"

#include "brdfs/glass.hpp"
#include "brdfs/lambertian.hpp"
#include "brdfs/metal.hpp"

#include "hitables/hitable_list.hpp"
#include "hitables/sphere.hpp"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

// TODO: implement miss object

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
              << file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    system("PAUSE");
    exit(99);
  }
}

// Save output buffer to .PNG file
int Save_PNG(float3 *fb, int width, int height, int samples) {
  stbi_flip_vertically_on_write(1);

  unsigned char *arr;
  arr = (unsigned char *)malloc(width * height * 3 * sizeof(unsigned char));

  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      int index = width * j + i;
      int pixel_index = 3 * (width * j + i);

      // average & gamma correct output color
      float3 col = sqrtf(fb[index] / float(samples));

      // Clamp and convert to [0, 255]
      col = 255.99f * clamp(col, 0.f, 1.f);

      // Copy int values to array
      arr[pixel_index + 0] = (int)col.x;  // R
      arr[pixel_index + 1] = (int)col.y;  // G
      arr[pixel_index + 2] = (int)col.z;  // B
    }
  }

  // Save .PNG file
  std::string fileName = "output.png";
  const char *name = (char *)fileName.c_str();
  return stbi_write_png(name, width, height, 3, arr, 0);
}

D_FUNCTION float3 color(const Ray &ray, Hitable **world, Miss **background,
                        uint &seed) {
  Ray cur_ray = ray;

  // float3 origin, direction;
  float3 throughput = make_float3(1.0f);

  for (int i = 0; i < 50; i++) {
    Hit_Record rec;

    // check if newly traced ray hits something
    if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
      Ray scattered;
      float3 attenuation;

      if (rec.brdf->scatter(cur_ray, rec, attenuation, scattered, seed)) {
        throughput *= attenuation;
        cur_ray = scattered;
      } else {
        return make_float3(0.f);
      }
    }

    // if not, ray missed. Light it with background color and return
    else {
      return throughput * (*background)->sample(cur_ray);
    }

    // origin = rec.hit_point;
    // direction = target - rec.hit_point;

    // Generate a new ray
    /*cur_ray = Ray(origin,      // new ray origin
                  direction);  // new ray direction*/
  }

  // Exceeded recursion
  return make_float3(0.f);
}

G_FUNCTION void create_world(Hitable **d_list, Hitable **d_world,
                             Camera **d_camera, Miss **d_miss, float width,
                             float height) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // get RNG seed
    uint seed = tea<64>(width, height);

    int i = 0;

    *d_miss =
        new Gradient_Background(make_float3(1.f),               // white
                                make_float3(0.5f, 0.7f, 1.f));  // light blue

    // ground sphere
    d_list[i++] = new Sphere(make_float3(0.f, -1000.f, 0.f), 1000.f,
                             new Lambertian(make_float3(0.5f)));

    for (int a = -11; a < 11; a++) {
      for (int b = -11; b < 11; b++) {
        float p = rnd(seed);

        // random center
        float3 center =
            make_float3(a + 0.9f * rnd(seed), 0.2f, b + 0.9f * rnd(seed));

        if (length(center - make_float3(4.f, 0.2, 0.f)) > 0.9) {
          // random color
          float3 color = make_float3(rnd(seed), rnd(seed), rnd(seed));

          if (p < (1.f / 3.f)) {
            d_list[i++] = new Sphere(center, 0.2f, new Lambertian(color));
          } else if (p < (2.f / 3.f)) {
            d_list[i++] =
                new Sphere(center, 0.2f, new Metal(color, 0.5f * rnd(seed)));
          } else {
            d_list[i++] =
                new Sphere(center, 0.2f, new Glass(make_float3(1.f), 1.5f));
          }
        }
      }
    }

    d_list[i++] = new Sphere(make_float3(0.f, 1.f, 0.f), 1.f,
                             new Glass(make_float3(1.f), 1.5f));
    d_list[i++] = new Sphere(make_float3(-4.f, 1.f, 0.f), 1.f,
                             new Lambertian(make_float3(0.4f, 0.2f, 0.1f)));
    d_list[i++] = new Sphere(make_float3(4, 1, 0), 1.f,
                             new Metal(make_float3(0.7f, 0.6f, 0.5f), 0.f));

    *d_world = new Hitable_List(d_list, i);

    const float3 lookfrom = make_float3(13.f, 2.f, 3.f);
    const float3 lookat = make_float3(0.f, 0.f, 0.f);

    *d_camera = new Camera(lookfrom,                      // lookfrom
                           lookat,                        // lookat
                           make_float3(0.f, 1.f, 0.f),    // upper axis
                           30.f,                          // vertical fov
                           float(width) / float(height),  // aspect ratio
                           0.1f,                          // aperture
                           10.f);                         // focus distance
  }
}

G_FUNCTION void free_world(Hitable **d_list, Hitable **d_world,
                           Camera **d_camera, Miss **d_miss) {
  (*d_world)->free();
  delete *d_world;
  delete *d_camera;
  delete *d_miss;
}

G_FUNCTION void render(float3 *fb, float3 *fb_acc,  // frame buffers
                       int width, int height,       // dimensions
                       int currentSample,           // current sample
                       int samples,                 // total number of samples
                       Camera **cam,                // camera object
                       Hitable **world,             // scene
                       Miss **background)           // background
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= width) || (j >= height)) return;

  int pixel_index = j * width + i;

  // get RNG seed
  uint seed = tea<64>(pixel_index, currentSample);

  // initialize acc buffer if needed
  if (currentSample == 0) fb_acc[pixel_index] = make_float3(0.f);

  float u = float(i + rnd(seed)) / float(width);
  float v = float(j + rnd(seed)) / float(height);

  // Trace a ray
  Ray ray = (*cam)->get_ray(u, v, seed);

  // save results
  fb_acc[pixel_index] += de_nan(color(ray, world, background, seed));
  fb[pixel_index] = fb_acc[pixel_index];
}

int main() {
  const int samples = 10;                   // number of samples
  const int W = 1200, H = 800;              // Image dimensions
  const int TX = 8, TY = 8;                 // Block dimensions
  size_t fb_size = W * H * sizeof(float3);  // frame buffer size

  // TODO: use one of these buffers to display the results on a GUI
  // allocate frame buffers
  float3 *fb;  // frame buffer for each sample
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
  float3 *fb_acc;  // frame buffer to accumulate samples
  checkCudaErrors(cudaMallocManaged((void **)&fb_acc, fb_size));

  // allocate and create world and camera
  Hitable **d_list;
  const int N_hitables = 22 * 22 + 1 + 3;  // TODO: get it dynamically
  checkCudaErrors(cudaMalloc((void **)&d_list, N_hitables * sizeof(Hitable *)));
  Hitable **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Hitable *)));
  Camera **d_camera;
  checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));
  Miss **d_miss;
  checkCudaErrors(cudaMalloc((void **)&d_miss, sizeof(Miss *)));
  create_world<<<1, 1>>>(d_list, d_world, d_camera, d_miss, W, H);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  clock_t start, stop;
  start = clock();

  // Render our buffer
  dim3 blocks(W / TX + 1, H / TY + 1);
  dim3 threads(TX, TY);

  for (int sample = 0; sample < samples; sample++) {
    render<<<blocks, threads>>>(fb, fb_acc, W, H, sample, samples, d_camera,
                                d_world, d_miss);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    printf("Render Progress: %.2f%%     \r", (sample * 100.f) / samples);
  }

  stop = clock();

  float timer_seconds = ((float)(stop - start)) / CLOCKS_PER_SEC;
  printf("Render Time: %.2f seconds\n", timer_seconds);

  // save image to file
  Save_PNG(fb, W, H, samples);

  // clean up
  checkCudaErrors(cudaDeviceSynchronize());
  free_world<<<1, 1>>>(d_list, d_world, d_camera, d_miss);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(d_camera));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_miss));
  checkCudaErrors(cudaFree(fb));
  checkCudaErrors(cudaFree(fb_acc));

  // useful for cuda-memcheck --leak-check full
  cudaDeviceReset();

  system("PAUSE");

  return 0;
}