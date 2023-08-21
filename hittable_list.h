#ifndef HITTABLELIST_H
#define HITTABLELIST_H
#include "hittable.h"
#include <memory>
class Hittable_list :public Hittable{
public:
	__device__  Hittable_list(){}
	__device__ Hittable_list(Hittable** list, int size) : list_size(size) { objects = list; }
 __device__ virtual bool hit(double t_min,double t_max,const Ray &r,Hit_record &hit_record)const  {
	 Hit_record tmp_hit_record;
		bool is_hit = false;
		double t_closest_so_far = t_max;
		for (int i = 0; i < list_size; i++) {
			if (objects[i]->hit(t_min,t_closest_so_far, r, tmp_hit_record)) { 
			is_hit = true;
	
			t_closest_so_far = tmp_hit_record.time;
			hit_record = tmp_hit_record;
			}
				
		}
		return is_hit;
	}
public:
Hittable ** objects;
	int list_size=0;
};
#endif