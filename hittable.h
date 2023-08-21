#pragma once
#ifndef HITTABLE_H
#define HITTABLE_H
#include "vec3.h"
#include "ray.h"
class Material;
struct Hit_record {
	double time;
	Vec3 hit_normal;
	Point3 hit_point;
	Material* mat;
	bool front_face;
public:
	__device__ void Set_normal(const Ray& r, const Point3 &outward_normal) {
		hit_normal = dot(r.dir(), outward_normal) < 0 ? outward_normal : -outward_normal;
		front_face = dot(r.dir(), outward_normal) < 0;
	}
};
class Hittable {
public:
	__device__ virtual bool hit(double t_min,double t_max,const Ray &r, Hit_record &hit_rec)const =0;


};
#endif