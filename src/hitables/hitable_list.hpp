#ifndef HITABLELISTH
#define HITABLELISTH

#include "../common.hpp"
#include "hitables.hpp"

class Hitable_List : public Hitable {
 public:
  __host__ __device__ Hitable_List() {}

  __host__ __device__ Hitable_List(Hitable** l, int n) {
    list = l;
    list_size = n;
  }

  __host__ __device__ virtual bool hit(const Ray& r, float tmin, float tmax,
                                       Hit_Record& rec) const;

  __host__ __device__ virtual void free() const {
    for (int i = 0; i < list_size; i++) {
      list[i]->free();
      delete list[i];
    }
  }

  Hitable** list;
  int list_size;
};

__host__ __device__ bool Hitable_List::hit(const Ray& r, float tmin, float tmax,
                                           Hit_Record& rec) const {
  Hit_Record temp_rec;

  bool hit_anything = false;
  float closest_so_far = tmax;

  // Check if a list element was hit
  for (int i = 0; i < list_size; i++) {
    if (list[i]->hit(r, tmin, closest_so_far, temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }

  return hit_anything;
}

class New_Hitable_List : public Hitable {
 public:
  __host__ __device__ New_Hitable_List() {}

  __host__ New_Hitable_List(int n) {
    cudaMallocManaged(&list, n * sizeof(Hitable*));
    cudaDeviceSynchronize();
  }

  __host__ __device__ virtual bool hit(const Ray& r, float tmin, float tmax,
                                       Hit_Record& rec) const;

  __host__ __device__ virtual void free() const {
    for (int i = 0; i < list_size; i++) {
      list[i].free();
      // delete list[i];
    }
  }

  Hitable* list;
  int list_size;
};

__host__ __device__ bool New_Hitable_List::hit(const Ray& r, float tmin,
                                               float tmax,
                                               Hit_Record& rec) const {
  Hit_Record temp_rec;

  bool hit_anything = false;
  float closest_so_far = tmax;

  // Check if a list element was hit
  for (int i = 0; i < list_size; i++) {
    if (list[i].hit(r, tmin, closest_so_far, temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }

  return hit_anything;
}

#endif