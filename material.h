#ifndef MATERIAL_H
#define MATERIAL_H
#include "cuda_runtime.h"
#include "ray.h"
#include "hittable.h"
#include "utility.h"
class Material {
public:
	__device__  virtual ~Material() = default;
	__device__ virtual bool scatter(const Ray& r_in, const Hit_record& hit_record, Color& attenuation, Ray& scattered,curandState *local_rand_state=nullptr)const = 0;
};
class Lambertian :public Material {
public:
	__device__ Lambertian(const Color &c):albedo(c) {}
	__device__ virtual bool scatter(const Ray& r_in, const Hit_record& hit_record, Color& attenuation, Ray& scattered,curandState *local_rand_state) const {
		Vec3 origin = hit_record.hit_point;
		Vec3 dir = hit_record.hit_normal + random_unit_vec(local_rand_state);
		attenuation = albedo;
		scattered = Ray(origin, dir);
		return true;
	}
private:
	Color albedo;
};
class Metal :public Material {
public:
	__device__ Metal(const Color &c,float f=0):albedo(c),fuzz(clamp(0,1,f)) {}
	__device__ virtual bool scatter(const Ray& r_in, const Hit_record& hit_record, Color& attenuation, Ray& scattered, curandState* local_curand_state = nullptr)const {
		Vec3 origin = hit_record.hit_point;
		Vec3 dir = reflect(unit_vec(r_in.dir()), hit_record.hit_normal)+fuzz*random_unit_vec(local_curand_state);
		attenuation = albedo;
		scattered = Ray(origin, dir);
		return dot(dir, hit_record.hit_normal) > 0;
	

	}
private:
	Color albedo;
	float fuzz;
};
class Dielectric :public Material {
public:
	__device__	Dielectric(float index_of_refraction) :ir(index_of_refraction) {

		}
	__device__ virtual bool scatter(const Ray& r_in, const Hit_record& hit_record, Color& attenuation, Ray& scattered, curandState* local_rand_state) const {
		attenuation = Color(1);
		float etai_over_etat = hit_record.front_face ? (1.0 / ir) : ir;
		float cos_theta = dot(-r_in.dir(),hit_record.hit_normal);
		float sin_theta = sqrt(1 - cos_theta * cos_theta);
		bool cannot_refract = etai_over_etat * sin_theta > 1.0f;
		
		if (cannot_refract || reflectance(cos_theta, etai_over_etat) > random_double(0, 1, local_rand_state)) {
			scattered = Ray(hit_record.hit_point, reflect(r_in.dir(), hit_record.hit_normal));
		}
		else {
			float sin_thetat = etai_over_etat * sin_theta;
			float cos_thetat = sqrt(1 - sin_thetat * sin_thetat);
			Vec3 u = unit_vec(r_in.dir() + dot(-r_in.dir(), hit_record.hit_normal) * hit_record.hit_normal);
			Vec3 dir = -hit_record.hit_normal * cos_thetat + u * sin_thetat;
			scattered = Ray(hit_record.hit_point, dir);
		}
		return true;
	}
private:
	__device__ static float reflectance(float cosine, float ref_idx) {
		auto r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow((1 - cosine), 5);
	}
	float ir;
};
#endif