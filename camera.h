#pragma once
#ifndef CAMERA_H
#define CAMERA_H
#include "vec3.h"
#include "ray.h"
#include "utility.h"
#include <cmath>
class Camera {
public:
	Camera() = default;

	__device__ __host__ Camera(const Point3 & lookfrom, const Vec3 & vup, const Point3 & lookat,double fov=20,double _focus_angle=10.0f,double dist_to_focus=3.4f,double aspect_ratio=16.0/9.0):origin(lookfrom),focus_angle(_focus_angle)
	{
		w = unit_vec(lookat - lookfrom);
		horizontal = unit_vec(cross(w, vup));
		vertical = cross( horizontal,w);
		double theta =degrees_to_radians( fov / 2);
		double h = tan(theta);
		viewport_height = 2 * h * dist_to_focus;
		viewport_width = viewport_height *aspect_ratio;
	 defocus_radius = dist_to_focus * tan(degrees_to_radians(_focus_angle / 2));
		u = horizontal;
		v = vertical;
		horizontal = horizontal * viewport_width;
		vertical = vertical * viewport_height;

		lower_left_corner = origin + dist_to_focus * w-vertical/2-horizontal/2;
		
	}
	__device__ Ray get_ray(double s, double t,curandState *curand_local_state)const  {
		Vec3 o = origin;
		if (focus_angle > 0) o = get_random_from_center(curand_local_state);
		return Ray(o, lower_left_corner + s * horizontal + t * vertical-o);


	}
private:
	Point3 lower_left_corner;
	Vec3 horizontal;
	Vec3 vertical;
	Vec3 w;
	Vec3 u, v;
	Vec3 origin;
	double viewport_width;
	double viewport_height;
	double focus_angle;
	float defocus_radius;
	__device__ Vec3 get_random_from_center(curandState *local_rand_state)const {
		auto p = defocus_radius*random_in_unit_disk(local_rand_state);
		p = p.x() * u + p.y() * v;
		return origin + p;

	}
};
#endif