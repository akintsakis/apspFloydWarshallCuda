# apspFloydWarshallCuda
CUDA implementation of the Floyd Warshall All pairs shortest path algorithm (blocked and regular version)

The program takes one argument, the size of the array. The float numbers within the array are randomly generated. When choosing the array size take into consideration available GPU memory.

The Floyd Warshall Algorithm is executed on the GPU first and then on the CPU so as to compare execution times. Generally, the GPU is orders of magnitude faster but it depends on the specific GPU and CPU used.

The blocked version uses a different algorithm that utilizes shared memory and is substantially faster than the regular version. 
