/* Kintsakis Athanasios AEM 6667 */

#define inf 9999
__global__ void funct(int n, int k, float* x, int* qx)
{
	
	int ix= blockIdx.x*blockDim.x + threadIdx.x;

		//Epeksigisi
		/*
		float temp2=x[i*n+k] + x[k*n+j];
		omws
		i=ix/n;		
		
		kai
		j=ix%n = ix&(n-1)
		i*n = ix/n * n = ix-ix%n= ix-j
				
		*/

		
		int j=ix&(n-1);
		float temp2=x[ix-j+k]+x[k*n+j];
		
		if(x[ix]>temp2)
		{
			x[ix]=temp2;
			qx[ix]=k;
		}
		
		if(x[ix]==inf)
		{
			qx[ix]=-2;
		}		
		
}
__global__ void funct2(int n, int k, float* x, int* qx)
{
	
	int ix= blockIdx.x*blockDim.x + threadIdx.x;

		int j=ix&(n-1);
		float temp2=x[ix-j+k]+x[k*n+j];
		if(x[ix]>temp2)
		{
			x[ix]=temp2;
			qx[ix]=k;
		}
		
}


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include  <time.h>
#include <sys/time.h>


int main(int argc, char **argv)
{

  struct timeval first, second, lapsed,third,fourth,lapsed2;
  struct timezone tzp,tzp2;
	
	
	float *host_A, *host_D;
	
	int *host_Q;
	float *dev_x;
	int *dev_qx;
	float *A;
	int *Q;
	float *D;

	int i,j,bk;
	int k=0;
	
	int n = atoi(argv[1]);
	
	printf("\n");
	printf("RUNNING WITH %d VERTICES \n",n);
	printf("\n");

	cudaMalloc(&dev_x, n*n*sizeof(float));
	cudaMalloc(&dev_qx, n*n*sizeof(float));
	

    //CPU arrays
	A=(float *)malloc(n*n*sizeof(float));  //arxikos pinakas A
	D=(float *)malloc(n*n*sizeof(float));  //arxikos pinakas D
	Q=(int *)malloc(n*n*sizeof(int));  //arxikos pinakas Q
	
	//GPU arrays
	host_A=(float *)malloc(n*n*sizeof(float));
	host_D=(float *)malloc(n*n*sizeof(float));
	host_Q=(int *)malloc(n*n*sizeof(int));
	


	
	srand(time(NULL));
	
	for(i=0;i<n;i++)
	{
		
		for(j=0;j<n;j++)
		{
			if(i==j)
			{A[i*n+j]=0;}
			else
			{
				A[i*n+j]=1200 * (float) rand()/RAND_MAX + 1;
				if(A[i*n+j]>1000)
				{
					A[i*n+j]=inf;
				}
			}
		}
	}
	
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			Q[i*n+j]=-1;
		}
	}

	
	for(i=0;i<n;i++)    
	{
		for(j=0;j<n;j++)
		{
			D[i*n+j]=A[i*n+j];
		}

	}

	
	for(i=0;i<n;i++)    //Antigrafi tou A ston host_A
	{
		for(j=0;j<n;j++)
		{
			host_A[i*n+j]=A[i*n+j];
		}

	}


	for(i=0;i<n;i++)    //Antigrafi tou Q ston host_Q
	{
		for(j=0;j<n;j++)
		{
			host_Q[i*n+j]=Q[i*n+j];
		}

	}
gettimeofday(&third, &tzp2);
////////////////////////////First Mem Copy////////////////////

 	gettimeofday(&first, &tzp);
	cudaMemcpy(dev_x, host_A, n*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_qx, host_Q, n*n*sizeof(int), cudaMemcpyHostToDevice);
	


gettimeofday(&second, &tzp);

  if(first.tv_usec>second.tv_usec){
    second.tv_usec += 1000000;
    second.tv_sec--;
  }
  
  lapsed.tv_usec = second.tv_usec - first.tv_usec;
  lapsed.tv_sec = second.tv_sec - first.tv_sec;

  printf("First Transfer CPU to GPU  Time elapsed: %lu, %lu s\n", lapsed.tv_sec, lapsed.tv_usec);
////////////////////////////////////////////////////GPU Calculation////////////////////////////////
	
	bk=(int)(n*n/512);
	int gputhreads=512;
	
	if(bk>0)
	{
		gputhreads=512;
	}
	else
	{
		bk=1;
		gputhreads=n*n;
	}	

	printf(" \n");
	printf("BLOCKS :   %d      GPU THREADS:     %d \n",bk,gputhreads);
	printf(" \n");	

	
gettimeofday(&first, &tzp);	
	
	funct<<<bk,gputhreads>>>(n,k,dev_x,dev_qx);
	
	for(k=1;k<n;k++)
	{		
		funct2<<<bk,gputhreads>>>(n,k,dev_x,dev_qx);
	}
	
cudaThreadSynchronize();

gettimeofday(&second, &tzp);

  if(first.tv_usec>second.tv_usec){
    second.tv_usec += 1000000;
    second.tv_sec--;
  }
  
  lapsed.tv_usec = second.tv_usec - first.tv_usec;
  lapsed.tv_sec = second.tv_sec - first.tv_sec;

  printf("GPU Calculation Time elapsed: %lu, %lu s\n", lapsed.tv_sec, lapsed.tv_usec);
  printf("\n");

//////////////////////////////////////////////////////////////////////////Second Mem Copy////////////////////
 gettimeofday(&first, &tzp);

cudaMemcpy(host_D, dev_x, n*n*sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(host_Q, dev_qx, n*n*sizeof(int), cudaMemcpyDeviceToHost);
gettimeofday(&second, &tzp);

  if(first.tv_usec>second.tv_usec){
    second.tv_usec += 1000000;
    second.tv_sec--;
  }
  
  lapsed.tv_usec = second.tv_usec - first.tv_usec;
  lapsed.tv_sec = second.tv_sec - first.tv_sec;

  printf("Second Transfer GPU to CPU  Time elapsed: %lu, %lu s\n", lapsed.tv_sec, lapsed.tv_usec);
  printf("\n");
//////////////////////////////////////////////////////////////////////

gettimeofday(&fourth, &tzp2); //total time

  if(third.tv_usec>fourth.tv_usec){
    fourth.tv_usec += 1000000;
    fourth.tv_sec--;
  }
  
  lapsed2.tv_usec = fourth.tv_usec - third.tv_usec;
  lapsed2.tv_sec = fourth.tv_sec - third.tv_sec;

  printf("TOTAL GPU + TRANSFERS  Time elapsed: %lu, %lu s\n", lapsed2.tv_sec, lapsed2.tv_usec);







//////////////////////////////////////////////////////////////


//CPU RUN 
printf(" THIS IS CPU \n");
printf("\n");
 gettimeofday(&first, &tzp);

	for(k=0;k<n;k++)
	{
		for(i=0;i<n;i++)
		{
			for(j=0;j<n;j++)
			{

				if((D[i*n+k]+D[k*n+j])<D[i*n+j])
				{
					D[i*n+j]=D[i*n+k]+D[k*n+j];
					Q[i*n+j]=k;					
				}
				if(D[i*n+j]==inf)
				{
					Q[i*n+j]=-2;
				}
			}
		}
	}
/////////////////////////////////////////////////////////////////
gettimeofday(&second, &tzp);

  if(first.tv_usec>second.tv_usec){
    second.tv_usec += 1000000;
    second.tv_sec--;
  }
  
  lapsed.tv_usec = second.tv_usec - first.tv_usec;
  lapsed.tv_sec = second.tv_sec - first.tv_sec;

  printf("CPU Time elapsed: %lu, %lu s\n", lapsed.tv_sec, lapsed.tv_usec);
/////////////////////////////////////////////////////

printf(" \n");

		
printf(" \n");

////////////////////////////////////////////////////////////////////////////////////////


/////////////FROM HERE AND UNDER ARE VALIDATION RUNS

printf("VALIDATING THAT D array from CPU and host_D array from GPU match... \n");


for(i=0;i<n;i++)
{
	for(j=0;j<n;j++)
	{
		if(abs(D[i*n+j]-host_D[i*n+j])>0.0001)
		{
			printf("ERROR MISMATCH in array D i %d j %d \n",i,j);
		}
	}
}
printf("OK \n");
printf("VALIDATING THAT Q array from CPU and host_Q match.... \n");


for(i=0;i<n;i++)
{
	for(j=0;j<n;j++)
	{
		if(Q[i*n+j]!=host_Q[i*n+j])
		{
			printf("ERROR MISMATCH in array Q i %d j %d \n",i,j);
		}
	}
}
printf("OK \n");
//printf("more validating .................... Patience!....\n");

printf("Initiating Path Rebuilding Check for CPU DATA... \n");
for(i=0;i<n;i++)
{
	for(j=0;j<n;j++)
	{

		if( (i==j) && (D[i*n+j]!=0) )
		{
			printf(" ERROR, i=j but element is not 0 \n");
		}

		if(D[i*n+j]==inf)
		{
			if(Q[i*n+j]!=-2)
			{
				printf("ERROR, No path exists between i %d and %d but array Q saysotherwise \n",i,j);
			}
			
		}
		else if(Q[i*n+j]==-1)
		{
			if(abs(A[i*n+j]-D[i*n+j])>0.0001)
			{
				printf("ERROR, from i %d to j %d cheapest route should be the direct one. D array not equal to array A cost \n",i,j);
			}
		}
		else
		{
			float temppath=0;
			int tempi=i;
			int tempj=j;
		
			while(Q[tempi*n+tempj]>=-1)
			{
				if(Q[tempi*n+tempj]==-1)
				{
					temppath=temppath+A[tempi*n+tempj];
					break;
				}
				temppath=temppath+D[tempi*n+Q[tempi*n+tempj]];
				tempi=Q[tempi*n+tempj];
				
			}

			if(abs(temppath-D[i*n+j])>0.0001)
			{
				printf("ERROR 2 in i %d j %d, expected %f and is %f \n",i,j,temppath,D[i*n+j]);
			}
		}
			
				
	}
}
printf("OK \n");
printf("Initiating Path Rebuilding Check for GPU DATA.... \n");

for(i=0;i<n;i++)
{
	for(j=0;j<n;j++)
	{

		if( (i==j) && (host_D[i*n+j]!=0) )
		{
			printf(" ERROR, i=j but element is not 0 \n");
		}

		if(host_D[i*n+j]==inf)
		{
			if(host_Q[i*n+j]!=-2)
			{
				printf("ERROR, No path exists between i %d and %d but array Q saysotherwise \n",i,j);
			}
			
		}
		else if(host_Q[i*n+j]==-1)
		{
			if(abs(host_A[i*n+j]-host_D[i*n+j])>0.0001)
			{
				printf("ERROR, from i %d to j %d cheapest route should be the direct one. host_D array not equal to host_A cost \n",i,j);
			}
		}
		else
		{
			float temppath=0;
			int tempi=i;
			int tempj=j;
		
			while(host_Q[tempi*n+tempj]>=-1)
			{
				if(host_Q[tempi*n+tempj]==-1)
				{
					temppath=temppath+host_A[tempi*n+tempj];
					break;
				}
				temppath=temppath+host_D[tempi*n+host_Q[tempi*n+tempj]];
				tempi=host_Q[tempi*n+tempj];
				
			}
				

			if(abs(temppath-host_D[i*n+j])>0.0001)
			{
				printf("Path REBUILT ERROR in cheching dinstance from in i %d j %d, expected %f and we found %f \n",i,j,temppath,host_D[i*n+j]);
			}
		}
			
				
	}
}
printf("OK \n");
printf("ALL OK WE ARE DONE \n");

return 0;

}
