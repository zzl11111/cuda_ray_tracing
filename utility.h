#pragma once
#ifndef UTILITY_H
#define UTILITY_H
#include <limits.h>
#include <cstdlib>
#include <curand_kernel.h>
#include "vec3.h"
constexpr const double pi = 3.1415926;
constexpr const double epsilon = 1e-6;
constexpr const double infinity = 1e+20;
__device__ __host__  double degrees_to_radians(double degree) {
	return degree*pi / 180;
}
__device__ __host__ float clamp(float min, float max, float num) {
	if (num < min)return min;
	if (num > max)return max;
	return num;

}
__device__ float random_double(float min, float max, curandState* local_rand_state) {
	return min + (max - min) * curand_uniform(local_rand_state);
}


__device__ Vec3 random_vec3(curandState* local_rand_state)
{
	return Vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
}
__device__ Vec3 random_vec3(float min, float max, curandState* local_rand_state) {
	return Vec3(random_double(min, max, local_rand_state), random_double(min, max, local_rand_state), random_double(min, max, local_rand_state));
}
__device__ Vec3 random_in_unit_sphere(curandState* local_rand_state) {
	auto p = 2 * random_vec3(local_rand_state) - Vec3(1);
	while (dot(p, p) > 1.0f) {
		p = 2 * random_vec3(local_rand_state) - Vec3(1);
	}
	return p;
}
__device__ Vec3 random_unit_vec(curandState* local_rand_state) {
	return unit_vec(random_in_unit_sphere(local_rand_state));
}
__device__ Vec3 random_in_unit_disk(curandState* local_rand_state) {
	while (true) {
		auto p = Vec3(random_double(-1, 1, local_rand_state), random_double(-1, 1, local_rand_state), 0);
		if (dot(p, p) < 1.0f)
		{
			return p;
		}
	}
}
#endif