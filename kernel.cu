
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include "vec3.h"
#include "ray.h"
#include "camera.h"
#include "hittable_list.h"
#include "sphere.h"
#include "material.h"
#include <curand_kernel.h>
#include<time.h>

//set the image_config
constexpr const int nx = 1600;
constexpr const int ny = 900;
constexpr const size_t num_pixels = nx * ny;
constexpr const int tx = 8;
constexpr const int ty = 8;
constexpr float aspect_ratio = (float)nx / (float)ny;
//set the camera config
Vec3 origin(13, 2, 3);
Vec3 lookat(0, 0, 0);
Vec3 lookup(0, 1, 0);

float defocus_angle = 0.6;
float focus_dist = 10.0f;
float fov = 20;

//define the cuda function

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
__device__ Color color( const Ray &r,Hittable **world,curandState* local_curand_state );
void check_cuda(cudaError_t result,  const char * const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}
__global__ void Rand_init(curandState* curand_state) {
    curand_init(1984, 0, 0, curand_state);
}
__global__ void render(double *fb,int max_x,int max_y,Camera* cam,Hittable **world,curandState *rand_state) {
    int i =threadIdx.x+blockIdx.x*blockDim.x ;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))return;
    curandState local_curand_state = rand_state[j * max_x + i];

    int pixel_index = j * max_x * 3 + i * 3;
    constexpr int total_ssp = 10;
    Color c(0,0,0);
    for (int ssp = 0; ssp < total_ssp; ssp++) {
        double u = (double(i)+curand_uniform(&local_curand_state)) / static_cast<float>(max_x);
        double v = (double(j)+curand_uniform(&local_curand_state)) / static_cast<float>(max_y);
        Ray r = cam->get_ray(u, v,&local_curand_state);
        c = c + color(r, world,& local_curand_state);
    }
    c = c / static_cast<float>(total_ssp);
    fb[pixel_index + 0] =sqrt(c.x());
    fb[pixel_index + 1] =sqrt(c.y());
    fb[pixel_index + 2] = sqrt(c.z());
}
__device__ Color  color(const  Ray& r, Hittable** world, curandState* local_curand_state) {

    Hit_record hit_record;
    constexpr int max_depth = 50;
    Ray cur_r = r;
    Color attenuation = Color(1);
    for (int i = 0; i < max_depth; i++)
    {
  
        if ((*world)->hit(0.001f, infinity, cur_r, hit_record))
        {
            Color tmp_attenuation;
            Ray scattered;
            if (hit_record.mat->scatter(cur_r, hit_record, tmp_attenuation, scattered, local_curand_state))
            {
                attenuation *= tmp_attenuation;
                cur_r = scattered;
            }

        }
        else {
            Vec3 unit_dir = unit_vec(cur_r.dir());
            float t = 0.5 * (unit_dir.y() + 1.0f);
            Color c=(1.0 - t)* Vec3(1, 1, 1) + t * Vec3(0.5, 0.7, 1.0);
            return c * attenuation;
        }
      
        }

    return Color(0, 0, 0);
}
__global__ void Create_scene(Hittable **objects ,Hittable ** list) {
    Lambertian* material_ground = new Lambertian(Color(0.8, 0.8, 0));
    Dielectric* material_left = new Dielectric(1.5);
    Lambertian* material_center = new Lambertian(Color(0.1, 0.2, 0.5));
    Metal* material_right = new Metal(Color(0.8, 0.6, 0.2), 0);
    *objects = new Sphere(Vec3(0,-100.5,-1),100,material_ground);
    *(objects + 1) = new Sphere(Vec3(-1, 0, -1), 0.5, material_left);
    *(objects + 2) = new Sphere(Vec3(-1, 0, -1), -0.4, material_left);
    *(objects + 3) = new Sphere(Vec3(0, 0, -1), 0.5, material_center);
    *(objects + 4) = new Sphere(Vec3(1, 0, -1), 0.5, material_right);
    *list = new Hittable_list(objects, 5);
}
__global__ void Free_world(Hittable** objects, Hittable** list,int num_objects) {
    for (int i = 0; i < num_objects; i++) {
        delete ((Sphere*)objects[i])->mat;
        delete* (objects+i);
    }
    delete* list;


}
__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= max_x || j >= max_y)return;
    int pixel_index = i + j * max_x;
    curand_init(1984,pixel_index,0,&rand_state[pixel_index]);


}
__global__ void create_random_scene(Hittable** world, Hittable** list,curandState *state) {
    Lambertian* ground = new Lambertian(Color(0.5, 0.5, 0.5));
    *world = new Sphere(Point3(0, -1000, 0),1000, ground);
    curandState local_rand_state = state[0];
    curandState* rand_state = &local_rand_state;
    int count = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            double choose_mat = random_double(0,1,rand_state);
            Point3 center(a + 0.9 * random_double(0, 1, rand_state), 0.2, b + 0.9 * random_double(0, 1, rand_state));
            if (choose_mat < 0.8) 
            {
                auto albedo = random_vec3(0, 1, rand_state) * random_vec3(0, 1, rand_state);
                Lambertian* sphere_material = new Lambertian(albedo);
                *(world+count) = new Sphere(center, 0.2, sphere_material);    
            }
            else if (choose_mat < 0.95) {
                auto albedo = random_vec3(0.5, 1, rand_state) * random_vec3(0.5, 1, rand_state);
                auto fuzz = random_double(0, 0.5, rand_state);
                Metal* sphere_material = new Metal(albedo, fuzz);
                *(world + count) = new Sphere(center, 0.2, sphere_material);
                
            }
            else {
                Dielectric* sphere_material = new Dielectric(1.5);
                *(world + count) = new Sphere(center, 0.2, sphere_material);
            }
            count++;
        }
       

    }
    Dielectric* material_1 = new Dielectric(1.5);
    *(world + count) = new Sphere(Point3(0, 1, 0), 1, material_1);
    count++; 
    Lambertian* material_2 = new Lambertian(Color(0.4, 0.2, 0.1));
    *(world + count) = new Sphere(Point3(-4, 1, 0), 1, material_2);
    count++;
    Metal* material_3 = new Metal(Color(0.7,0.6,0.5),0);
    *(world + count) = new Sphere(Point3(4, 1, 0), 1, material_3);
    count++;
    *list = new Hittable_list(world, count);
}



int main()
{
    int num_objects=488;

    double * fb;
    dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
    dim3 threads(tx, ty);
    checkCudaErrors(cudaMallocManaged((void**)&fb, num_pixels * sizeof(double) * 3));
    Camera cam(origin, lookup, lookat,fov,defocus_angle,focus_dist,aspect_ratio);
    Camera* cam_ptr;
    Hittable** world;
    Hittable** world_list;
    curandState* d_state;
    checkCudaErrors(cudaMalloc((void**)&world, num_objects * sizeof(Hittable*)));
    checkCudaErrors(cudaMalloc((void**)&world_list,  sizeof(Hittable*)));
    checkCudaErrors(cudaMalloc((void**)&d_state, num_pixels * sizeof(curandState)));
    render_init << <blocks, threads >> > (nx, ny, d_state);
    //Create_scene << <1, 1 >> > (world, world_list);
   create_random_scene<<<1, 1 >>> (world,world_list,d_state);
    clock_t start, end;
    start = clock();

    cudaDeviceSynchronize();
    cudaMalloc((void **)&cam_ptr, sizeof(Camera));
    cudaMemcpy(cam_ptr, &cam, sizeof(Camera), cudaMemcpyHostToDevice);
    render<<<blocks, threads>>> (fb,nx,ny,cam_ptr,world_list,d_state);
    checkCudaErrors(cudaDeviceSynchronize());
    end = clock();
    double time_seconds = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cerr << "took " << time_seconds << "\n";
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
#pragma omp for 
    for (int j = ny - 1; j >= 0; j--) {
        std::cerr  << j << "lines remain\n"<<std::flush;
        for (int i = 0; i < nx; i++) {
            auto index = (i + j * nx) * 3;

            double r = fb[index  + 0];
            double  g = fb[index  + 1];
            double b = fb[index  + 2];
            int ir = static_cast<int>(clamp(0,255,r*255.99)); 
            
            int ig = static_cast<int>(clamp(0,255,g * 255.99));
            int ib = static_cast<int>(clamp(0,255,b * 255.99));
            std::cout << ir << " " << ig << " " << ib<<"\n";
        }
    }
    Free_world << <1, 1 >> > (world,world_list,num_objects);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.