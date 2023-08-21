#ifndef VEC3_H
#define VEC3_H
#include "cuda_runtime.h"
#include <cmath>
using std::sqrt;
class Vec3 {
public:
	__host__ __device__ Vec3() {}
	__host__ __device__ Vec3(float x, float y, float z) { e[0] = x; e[1] = y; e[2] = z; }
	__host__ __device__ float x()const  { return e[0]; }
	__host__ __device__ float y()const  { return e[1]; }
	__host__ __device__ float z() const { return e[2]; }
	__host__ __device__ Vec3(const Vec3& v) :Vec3(v.x(),v.y(),v.z()) {}
	__host__ __device__ Vec3(float t):Vec3(t,t,t){}
	__host__ __device__ float operator[](int i) { return e[i]; }
	__host__ __device__ Vec3 operator-()const { return Vec3(-e[0], -e[1], -e[2]); }
	__host__ __device__ Vec3 operator*(const Vec3& v)const { return Vec3(e[0] * v.x(), e[1] * v.y(), e[2] * v.z()); }
	__host__ __device__ Vec3 operator*(float t)const { return Vec3( e[0] * t, e[1] * t, e[2] * t); }
	__host__ __device__ Vec3& operator*=(const Vec3& v) 
	{
		*this = *this * v;
		return *this ;
	}
	__host__ __device__ Vec3& operator *=(float t) {
		*this = *this * t;
		return *this;
	}
	__host__ __device__ Vec3 operator+(const Vec3& v)const { return Vec3(v.x() + e[0], v.y() + e[1], v.z() + e[2]); }
	__host__ __device__ Vec3 operator-(const Vec3& v)const {return Vec3(e[0]-v.x(),e[1]-v.y(),e[2]-v.z()); }
	__host__ __device__ __inline__ Vec3 operator/(float t)const { return *this * (1 / t); }
	
	__host__ __device__ bool near_zero()const
	{
		auto epsilon = 1e-8;
		return (fabs(e[0]) < epsilon && fabs(e[1]) < epsilon && fabs(e[2]) < epsilon);
	}
private:
	float e[3];
};
inline std::ostream& operator<<(std::ostream& out, const Vec3& v) {
	out << v.x() << " " << v.y() << " " << v.z();
	return out;
}
__host__ __device__ __inline__ Vec3 operator*(float t, const Vec3& v) { return v * t; }
__host__ __device__ Vec3 unit_vec(const Vec3& v) {
	float norm = sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
	return Vec3(v.x() / norm, v.y() / norm, v.z() / norm);
}
__host__ __device__ Vec3 cross(const Vec3& v1, const Vec3& v2) {
	return Vec3(v1.y() * v2.z() - v2.y() * v1.z(),v1.z()*v2.x()-v1.x()*v2.z(),v1.x()*v2.y()-v1.y()*v2.x() );
}
__host__ __device__ inline float dot(const Vec3& v1, const Vec3& v2) {
	return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
}
__host__ __device__ Vec3 reflect(const Vec3& v, const Vec3& n) {
	return v - 2 * dot(v, unit_vec(n)) * unit_vec(n);
}

using Point3 = Vec3;
using Color = Vec3;
#endif