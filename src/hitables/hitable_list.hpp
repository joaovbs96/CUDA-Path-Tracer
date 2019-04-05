#ifndef HITABLELISTH
#define HITABLELISTH

#include <vector>
#include "../common.hpp"
#include "hitables.hpp"

class Hitable_List : public Hitable {
  const size_t length;
  Hitable** begin;

 public:
  __host__ Hitable_List(int _length, std::vector<Hitable*> array, int index = 0)
      : length(_length), Hitable(index) {
    cudaMallocManaged(&begin, length * sizeof(Hitable*));
    cudaDeviceSynchronize();

    //std::copy(array.begin(), array.end(), begin);
    memcpy((void *)begin, (void *)array.data(), length * sizeof(Hitable*));
  }

  __host__ Hitable_List(int _length, Hitable** arr, int index = 0)
      : length(_length), Hitable(index) {
    cudaMallocManaged(&begin, length * sizeof(Hitable*));
    cudaDeviceSynchronize();

    /*for(int i = 0; i < length; i++) {
      begin[i] = arr[i];
    }*/

    // memcpy((void *)begin, (void *)arr, length * sizeof(Hitable*));
  }

  __host__ Hitable_List(int _length, int index = 0)
      : length(_length), Hitable(index) {
    cudaMallocManaged(&begin, length * sizeof(Hitable*));
    cudaDeviceSynchronize();
  }

  __host__ __device__ ~Hitable_List() {
    cudaDeviceSynchronize();
    cudaFree(begin);
  }

  __host__ __device__ virtual int sum() const {
    int temp = 0;
    
    for (int i = 0; i < length; i++) {
      temp += (*begin + index)->index;
    }

    return temp;
  }

  Hitable_List(const Hitable_List& uarray, int index = 0)
      : length(uarray.length), Hitable(index) {
    cudaMallocManaged(&begin, length * sizeof(Hitable*));
    memcpy((void*)begin, (void*)uarray.begin, length * sizeof(Hitable*));
  }

  __host__ __device__ virtual void free() const {}

  __host__ __device__ bool hit(const Ray& r, float tmin, float tmax,
                               Hit_Record& rec) const {
    Hit_Record temp_rec;

    bool hit_anything = false;
    float closest_so_far = tmax;

    // Check if a list element was hit
    for (int i = 0; i < length; i++) {
      if (begin[i]->hit(r, tmin, closest_so_far, temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    }

    return hit_anything;
  }
};

class Old_Hitable_List : public Hitable {
 public:
  __host__ __device__ Old_Hitable_List() : Hitable(0) {}

  __host__ __device__ Old_Hitable_List(Hitable** l, int n, int index = 0)
      : Hitable(index) {
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

__host__ __device__ bool Old_Hitable_List::hit(const Ray& r, float tmin,
                                               float tmax,
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

#endif