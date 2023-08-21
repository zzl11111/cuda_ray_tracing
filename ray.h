#ifndef RAY_H
#define RAY_H
#include "vec3.h"
class Ray {
public:
	__device__ Ray():origin(Vec3(0,0,0)),direction(0,0,-1) {}
	__device__ Ray(const Vec3 &o,const Vec3 &dir):origin(o),direction(unit_vec(dir)) {}
	__device__ Point3 o() const { return origin; }
	__device__ Vec3 dir()const { return direction; }
	__device__ Point3 at(double t)const { return origin + t * direction; }
public:
	Point3 origin;
	Vec3 direction;
};
#endif