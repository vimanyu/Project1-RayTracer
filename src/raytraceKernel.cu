// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  
  ray r;
  float theta = fov.x*PI/180.0f;
  float phi = fov.y*PI/180.0f;

  glm::vec3 A = glm::cross(view,up);
  glm::vec3 B = glm::cross(A,view);
  glm::vec3 M = eye + view;
  glm::vec3 H = glm::normalize(A)*glm::length(view)*tan(theta);
  glm::vec3 V = glm::normalize(B)*glm::length(view)*tan(phi);

  float sx = (float)x/(resolution.x-1);
  float sy = 1.0 - (float)y/(resolution.y-1);

  glm::vec3 P = M + (2*sx-1)*H + (2*sy - 1)*V;
  r.origin = eye;
  r.direction = glm::normalize(P-eye);
  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){
      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}


__host__ __device__ int findNearestGeometricIntersection(ray& r, glm::vec3& intersectionPoint,
														   glm::vec3& intersectionNormal,
														   staticGeom* geoms, int numberOfGeoms)
{
	int nearestIntersectionObject = -1;
	float nearestIntersectionDist = FLT_MAX;
	for(int i=0; i<numberOfGeoms;++i)
	{
		if(geoms[i].type == SPHERE)
		{
			glm::vec3 iPoint;
			glm::vec3 iNormal;		
			float t = sphereIntersectionTest(geoms[i],r,iPoint,iNormal);
			if (t!= -1 && t<nearestIntersectionDist)
			{
				nearestIntersectionObject = i;
				nearestIntersectionDist = t;
				intersectionPoint = iPoint;
				intersectionNormal = iNormal;
			}
		}

		if(geoms[i].type == CUBE)
		{
			glm::vec3 iPoint;
			glm::vec3 iNormal;		
			float t = boxIntersectionTest(geoms[i],r,iPoint,iNormal);
			if (t!= -1 && t<nearestIntersectionDist)
			{
				nearestIntersectionObject = i;
				nearestIntersectionDist = t;
				intersectionPoint = iPoint;
				intersectionNormal = iNormal;
			}
		}		
	}

	return nearestIntersectionObject;


}

__host__ __device__ glm::vec3 shade(material& mtl, glm::vec3& shadePoint, glm::vec3& shadeNormal,
									glm::vec3 eye,staticGeom* geoms,int numberOfGeoms)
{
	glm::vec3 lightPos( 0,10,20);
	glm::vec3 lightCol(1,1,1);
	int numberOfLights = 1;
	float Kd = 0.6;
	float Ks = 0.2;
	glm::vec3 color(0,0,0);
	
	for (int i=0;i<numberOfLights;++i)
	{
		ray shadowFeeler;
		shadowFeeler.direction = lightPos - shadePoint;
		shadowFeeler.origin = shadePoint+ (float)RAY_BIAS_AMOUNT*shadowFeeler.direction;
		
		glm::vec3 intersectionPoint,intersectionNormal;
		int intersectionObjIndex = findNearestGeometricIntersection(shadowFeeler,intersectionPoint,
																	intersectionNormal,
																	geoms,numberOfGeoms);

		if (intersectionObjIndex != -1)
			continue;


		float LN = glm::dot(shadowFeeler.direction,shadeNormal);
		LN = utilityCore::clamp(LN,0,1);
		glm::vec3 Rj = glm::normalize(glm::reflect(-shadowFeeler.direction,shadeNormal));
		glm::vec3 V = glm::normalize(eye-shadePoint);
		float RjV = glm::dot(Rj,V);
		RjV = utilityCore::clamp(RjV,0,1);

		color+= (Kd*mtl.color*LN + Ks*mtl.specularColor*(powf(RjV,mtl.specularExponent)));
	}

	return color;
}



//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* mtls, int numberOfMaterials,
							light* lights,int numberOfLights){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y)){

	
	ray r = raycastFromCameraKernel(resolution,time,x,y,cam.position,cam.view,cam.up,cam.fov);

	colors[index] = glm::vec3(1,0,0);

	glm::vec3 intersectionPoint;
	glm::vec3 intersectionNormal;
	int nearestIntersectionObject = -1;
	float nearestIntersectionDist = FLT_MAX;
	for(int i=0; i<numberOfGeoms;++i)
	{
		if(geoms[i].type == SPHERE)
		{
			glm::vec3 iPoint;
			glm::vec3 iNormal;		
			float t = sphereIntersectionTest(geoms[i],r,iPoint,iNormal);
			if (t!= -1 && t<nearestIntersectionDist)
			{
				nearestIntersectionObject = i;
				nearestIntersectionDist = t;
				intersectionPoint = iPoint;
				intersectionNormal = iNormal;
			}
		}

		else if(geoms[i].type == CUBE)
		{
			glm::vec3 iPoint;
			glm::vec3 iNormal;		
			float t = boxIntersectionTest(geoms[i],r,iPoint,iNormal);
			if (t!= -1 && t<nearestIntersectionDist)
			{
				nearestIntersectionObject = i;
				nearestIntersectionDist = t;
				intersectionPoint = iPoint;
				intersectionNormal = iNormal;
			}
		}		
	}

	material mtl = mtls[geoms[nearestIntersectionObject].materialid];

	glm::vec3 color(0,0,0);
	
	for (int i=0;i<numberOfLights;++i)
	{
		ray shadowFeeler;
		glm::vec3 lightPos = lights[i].position;
		glm::vec3 lightCol = lights[i].color;
		float lightIntensity = lights[i].intensity;
		glm::vec3 ptToLight = lightPos-intersectionPoint;
		shadowFeeler.direction = glm::normalize(ptToLight);
		shadowFeeler.origin = intersectionPoint+ (float)RAY_BIAS_AMOUNT*shadowFeeler.direction;
		bool occluded = false;
		
		float distSquared = ptToLight.x*ptToLight.x + 
							ptToLight.y*ptToLight.y +
							ptToLight.z*ptToLight.z;

		for(int i=0; i<numberOfGeoms;++i)
		{
			if(geoms[i].type == SPHERE)
			{
				glm::vec3 iPoint;
				glm::vec3 iNormal;		
				float t = sphereIntersectionTest(geoms[i],shadowFeeler,iPoint,iNormal);

				if (t!= -1)
				{
					glm::vec3 intersectionPoint = getPointOnRay(shadowFeeler,t);
					glm::vec3 ptToIntersection = intersectionPoint - shadowFeeler.origin;
					float dsq = ptToIntersection.x*ptToIntersection.x + 
					ptToIntersection.y*ptToIntersection.y +
					ptToIntersection.z*ptToIntersection.z;
					if (dsq<distSquared)
					{
			    		occluded = true;
						break;
					}
				}
			}

			if(geoms[i].type == CUBE)
			{
				glm::vec3 iPoint;
				glm::vec3 iNormal;		
				float t = boxIntersectionTest(geoms[i],shadowFeeler,iPoint,iNormal);

				if (t!= -1)
				{
					glm::vec3 intersectionPoint = getPointOnRay(shadowFeeler,t);
					glm::vec3 ptToIntersection = intersectionPoint - shadowFeeler.origin;
					float dsq = ptToIntersection.x*ptToIntersection.x + 
					ptToIntersection.y*ptToIntersection.y +
					ptToIntersection.z*ptToIntersection.z;
					if (dsq<distSquared)
					{
			    		occluded = true;
						break;
					}
				}
			}
		
		}

		if(occluded)
			continue;

		float LN = glm::dot(shadowFeeler.direction,intersectionNormal);
		LN = max(LN,0.0f);
		LN = min(LN,1.0f);
		glm::vec3 reflect = -shadowFeeler.direction-2.0f*intersectionNormal*
													glm::dot(-shadowFeeler.direction,intersectionNormal);
		glm::vec3 Rj = glm::normalize(reflect);
		glm::vec3 V = glm::normalize(cam.position-intersectionPoint);
		float RjV = glm::dot(Rj,V);
		RjV = max(RjV,0.0f);
		RjV = min(RjV,1.0f);

		//color+= Kd*mtl.color*LN;
		//color += glm::vec3(shadowFeeler.direction);
		//color+= glm::vec3( intersectionNormal.x,intersectionNormal.y,intersectionNormal.z);
		//color+= glm::vec3( fabs(intersectionNormal.x),fabs(intersectionNormal.y),fabs(intersectionNormal.z));
		//color+= glm::vec3(LN,LN,LN);
		glm::vec3 diffColor = mtl.diffuseCoefficient*lightIntensity*LN*mtl.color;
		glm::vec3 specColor = mtl.specularCoefficient*mtl.specularColor*(powf(RjV,mtl.specularExponent));
		color+= (diffColor+specColor);
	}	
	colors[index] = color;

   }
}




//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms,light* lights, int numberOfLights){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  //package materials
  material* cudamtls = NULL;
  cudaMalloc( (void**)&cudamtls, numberOfMaterials*sizeof(material));
  cudaMemcpy(cudamtls,materials,numberOfMaterials*sizeof(material),cudaMemcpyHostToDevice);

  //package lights
  light* cudalights = NULL;
  cudaMalloc( (void**)&cudalights, numberOfLights*sizeof(light));
  cudaMemcpy(cudalights,lights,numberOfLights*sizeof(light),cudaMemcpyHostToDevice);

  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms,cudamtls,numberOfMaterials,cudalights,numberOfLights);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree(cudamtls);
  cudaFree(cudalights);
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
