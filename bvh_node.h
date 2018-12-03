#ifndef BVHNODEH
#define BVHNODEH

#include "hitable.h"

class bvh_node : public hitable {
public:
	__device__ bvh_node() {}

	__device__ bvh_node(hitable **l, int n, float time0, float time1, curandState *local_rand_state);

	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;

	__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;

	__device__ virtual hitableType getType() const {
		return BVH_NODE;
	}

	hitable *left, *right;
	aabb box;
};

__device__ bool bvh_node::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
	printf("%d,%d\n", left->getType(), right->getType());
	if (box.hit(r, tmin, tmax)) {
		hit_record left_rec, right_rec;

		bool hit_left = left->hit(r, tmin, tmax, left_rec);
		bool hit_right = right->hit(r, tmin, tmax, right_rec);

		if (hit_left && hit_right) {
			rec = left_rec.t < right_rec.t ? left_rec : right_rec;
			return true;
		}
		else if (hit_left) {
			rec = left_rec;
			return true;
		}
		else if (hit_right) {
			rec = right_rec;
			return true;
		}
		else
			return false;
	}

	return false;
}

__device__ bool bvh_node::bounding_box(float t0, float t1, aabb& b) const {
	b = box;
	return true;
}

__device__ int box_x_compare(hitable * a, hitable * b) {
	aabb box_left, box_right;

	if (!a->bounding_box(0, 0, box_left) || !b->bounding_box(0, 0, box_right)) {
		printf("no bounding box in bvh_node constructor\n");
	}

	if (box_left.min().x() - box_right.min().x() < 0.0)
		return -1;
	else
		return 1;
}

__device__ int box_y_compare(hitable * a, hitable * b) {
	aabb box_left, box_right;

	if (!a->bounding_box(0, 0, box_left) || !b->bounding_box(0, 0, box_right)) {
		printf("no bounding box in bvh_node constructor\n");
	}

	if (box_left.min().y() - box_right.min().y() < 0.0)
		return -1;
	else
		return 1;
}

__device__ int box_z_compare(hitable * a, hitable * b) {
	aabb box_left, box_right;

	if (!a->bounding_box(0, 0, box_left) || !b->bounding_box(0, 0, box_right)) {
		printf("no bounding box in bvh_node constructor\n");
	}

	if (box_left.min().z() - box_right.min().z() < 0.0)
		return -1;
	else
		return 1;
}

__device__ void myswap(hitable **a, hitable **b) {
	hitable *temp;
	temp = *a;
	*a = *b;
	*b = temp;
}

__device__ void mysort(hitable **l, int n, int axis) {
	for (int i = 0; i < n - 1; i++) {
		bool is_sorted = true;

		for (int j = 0; j < n - 1; j++) {
			if (axis == 0) {
				if (box_x_compare(l[j], l[j + 1]) < 0) {
					myswap(&l[j], &l[j + 1]);
					is_sorted = false;
				}
			}
			else if (axis == 1) {
				if (box_y_compare(l[j], l[j + 1]) < 0) {
					myswap(&l[j], &l[j + 1]);
					is_sorted = false;
				}
			}
			else {
				if (box_z_compare(l[j], l[j + 1]) < 0) {
					myswap(&l[j], &l[j + 1]);
					is_sorted = false;
				}
			}
		}

		if (is_sorted)
			return;
	}
}

__device__ inline  bvh_node::bvh_node(hitable **l, int n, float time0, float time1, curandState *local_rand_state) {
	if (n == 0)
		return;

	if (n > 1) {
		int axis = int(3 * curand_uniform(local_rand_state));
		mysort(l, n, axis);
	}

	printf("n: %d\n", n);
	if (n <= 1) {
		left = right = l[0];
	}
	else if (n == 2) {
		left = l[0];
		right = l[1];
	}
	else {
		left = new bvh_node(l, n / 2, time0, time1, local_rand_state);
		right = new bvh_node(l + (n / 2), n - (n / 2), time0, time1, local_rand_state);
	}

	aabb box_left, box_right;
	if (!left->bounding_box(time0, time1, box_left) || !right->bounding_box(time0, time1, box_right))
		printf("no bounding box in bvh_node constructor\n");

	box = surrounding_box(box_left, box_right);
}

/*__device__ float bvh_node::pdf_value(const vec3& o, const vec3& v) const {
	return (left->pdf_value(o, v) + right->pdf_value(o, v)) / 2;
}

__device__ vec3 bvh_node::random(const vec3& o) const {
	if (RFG() > 0.5) // right
		return right->random(o);
	else // left
		return left->random(o);
}*/

#endif // !BVHNODEH
