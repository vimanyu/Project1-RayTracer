-------------------------------------------------------------------------------
CUDA Raytracer
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
This project implements a ray tracer using the awesome parallelization abilities
of the GPU using CUDA. Ray tracing is by itself an embarrasingly parallel 
algorithm, each pixel can be processed independently of the other pixels. 

-------------------------------------------------------------------------------
FEATURES:
-------------------------------------------------------------------------------

* Parse a scene description format (TAKUAscene as described below)
* Support for multiple point lights
* Phong shading model


-------------------------------------------------------------------------------
TAKUAscene FORMAT:
-------------------------------------------------------------------------------
This project uses a custom scene description format, called TAKUAscene.
TAKUAscene files are flat text files that describe all geometry, materials,
lights, cameras, render settings, and animation frames inside of the scene.
Items in the format are delimited by new lines, and comments can be added at
the end of each line preceded with a double-slash.

Materials are defined in the following fashion:

* MATERIAL (material ID)								//material header
* RGB (float r) (float g) (float b)					//diffuse color
* SPECX (float specx)									//specular exponent
* SPECRGB (float r) (float g) (float b)				//specular color
* REFL (bool refl)									//reflectivity flag, 0 for
  no, 1 for yes
* REFR (bool refr)									//refractivity flag, 0 for
  no, 1 for yes
* REFRIOR (float ior)									//index of refraction
  for Fresnel effects
* SCATTER (float scatter)								//scatter flag, 0 for
  no, 1 for yes
* ABSCOEFF (float r) (float b) (float g)				//absorption
  coefficient for scattering
* RSCTCOEFF (float rsctcoeff)							//reduced scattering
  coefficient
* EMITTANCE (float emittance)							//the emittance of the
  material. Anything >0 makes the material a light source.

Cameras are defined in the following fashion:

* CAMERA 												//camera header
* RES (float x) (float y)								//resolution
* FOVY (float fovy)										//vertical field of
  view half-angle. the horizonal angle is calculated from this and the
  reslution
* ITERATIONS (float interations)							//how many
  iterations to refine the image, only relevant for supersampled antialiasing,
  depth of field, area lights, and other distributed raytracing applications
* FILE (string filename)									//file to output
  render to upon completion
* frame (frame number)									//start of a frame
* EYE (float x) (float y) (float z)						//camera's position in
  worldspace
* VIEW (float x) (float y) (float z)						//camera's view
  direction
* UP (float x) (float y) (float z)						//camera's up vector

Objects are defined in the following fashion:
* OBJECT (object ID)										//object header
* (cube OR sphere OR mesh)								//type of object, can
  be either "cube", "sphere", or "mesh". Note that cubes and spheres are unit
  sized and centered at the origin.
* material (material ID)									//material to
  assign this object
* frame (frame number)									//start of a frame
* TRANS (float transx) (float transy) (float transz)		//translation
* ROTAT (float rotationx) (float rotationy) (float rotationz)		//rotation
* SCALE (float scalex) (float scaley) (float scalez)		//scale

An example TAKUAscene file setting up two frames inside of a Cornell Box can be
found in the scenes/ directory.

-------------------------------------------------------------------------------
Development process with screenshots:
-------------------------------------------------------------------------------

The following renders were done to check the correctness of the code and various
routines

* Test the code that casts rays
![alt tag](https://raw.github.com/vimanyu/Project1-RayTracer/master/renders/test_ray_shooting.bmp)

* Test the intersection tests code
![alt tag](https://raw.github.com/vimanyu/Project1-RayTracer/master/renders/test_intersection_normals.bmp)

* Test diffuse falloffs ( dot product of light direction and normals)
![alt tag](https://raw.github.com/vimanyu/Project1-RayTracer/master/renders/diffuse_dot_products.bmp)

* Test specular highlights
![alt tag](https://raw.github.com/vimanyu/Project1-RayTracer/master/renders/specular_highlights.bmp)

* Phong shading
![alt tag](https://raw.github.com/vimanyu/Project1-RayTracer/master/renders/phong_shading.bmp)

* Supersampling and reflections
![alt tag](https://raw.github.com/vimanyu/Project1-RayTracer/master/renders/supersampled_and_reflections.bmp)

