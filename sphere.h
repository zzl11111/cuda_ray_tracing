#ifndef SPHERE_H
#define SPHERE_H
#include "hittable.h"
class Sphere :public Hittable {

public:
	__device__ Sphere() {

	}
	__device__  Sphere(const Point3& c, double r,Material *_mat) 
		:center(c),radius(r),mat(_mat)
	{

	}
__device__ virtual bool hit(double t_min,double t_max,const Ray &r,Hit_record & hit_record)const {

	Vec3 oc = r.o() - center;
	Vec3 dir = r.dir();
	double a = dot(dir, dir);
	double b = 2 * dot(oc, dir);
	double c = dot(oc, oc) - radius * radius;
	double delta = b * b - 4 * a * c;
	
	if (delta < 0) {
		return false;
	}
	
	double t = (-b - std::sqrt(delta)) / (2 * a);
	if (t<t_min || t>t_max) {
		t = (-b + std::sqrt(delta)) / (2 * a);
		if (t<t_min || t>t_max) {
			return false;
		}
	}
	hit_record.time = t;

	hit_record.hit_point = r.at(t);
	hit_record.mat = mat;
	Vec3 outward_normal = (r.at(t) - center)/radius;
	hit_record.Set_normal(r, outward_normal);
	return true;




	}
private:
	Point3 center;
	double radius;
public:

	Material* mat;

};

#endif