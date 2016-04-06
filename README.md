# apspFloydWarshallCuda
CUDA implementation of the Floyd Warshall All pairs shortest path algorithm (blocked and regular version)

The program takes one argument, the size of the array. The float numbers within the array are randomly generated. When choosing the array size take into consideration available GPU memory.

The Floyd Warshall Algorithm is executed on the GPU first and then on the CPU so as to compare execution times. Generally, the GPU is orders of magnitude faster but it depends on the specific GPU and CPU used.

The blocked version uses a different algorithm that utilizes shared memory and is substantially faster (almost twice as fast) than the regular version. 
A good CUDA introductory example showcasing efficent shared memory usage. 

To compile: 
nvcc <name.cu> -o <name.out>
nvcc <name.cu> -o <name.out> -arch compute_20 -code compute_20 (specifying arch and code, use whatever values your card supports)

To run:
./<name.out> <number_of_vertices>
Example: ./name.out 256

Athanassios Kintsakis
athanassios.kintsakis@gmail.com
akintsakis@issel.ee.auth.gr
