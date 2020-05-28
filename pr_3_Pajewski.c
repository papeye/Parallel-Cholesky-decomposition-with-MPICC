#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/*This code is parallel implementation of Cholesky decomposition of a symmetric matrix. It works for any rank=n.
Calcultions are done sequentionally column after column (only diagonal and below diagonal elements are calculated). Program is meant to 
be compiled with MPICC: unix command "mpicc cholesky.c -o cholesky -std=c99 -lm" will produce file "cholesky" which is meant to be run with n or more
processors.
*/
int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);
  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  //Given below matrix is a 4x4 example but program works for any rank symmetric matrix 
  int n=5; //rank of the matrix
  double M[5][5] = {{81, 9, 36, 18, 9}, {9, 5, 14, 8, 5}, {36, 14, 50, 32, 23}, {18, 8, 32, 71, 45}, {9, 5, 23, 45, 55}}; //declaration of the matrix to be decomposed
  for (int k=0; k<n; k++)//k-column index (algorithm goes below of diagonal from left to right)
  {		  
	if(world_rank == 0){
		//printing to screen initial matrix M
		if (k==0)
		{	
			printf("Matrix M=\n");
			for (int i = 0; i <n; i++){
				for (int j = 0; j <n; j++){
					printf("%f", M[i][j]);
					if(j!=n-1)
					{
						printf(", ");	
					}	
				}
				printf("\n");
			}			
		}	
		//0 procesor calculates every diagonal element
		for (int j=0; j<k; j++)
		{
			M[k][k]-=(M[k][j]*M[k][j]);	
		}
		M[k][k]=sqrt(M[k][k]);
		
		for (int p=1; p<n-k; p++) //p-processor index to which 0 processor sends a given task
		{	
			for (int c=0; c<=k; c++)//processor 0 sends already calculated values
			{
				for (int r=c; r<n; r++)
				{	
					MPI_Send(&M[r][c], 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
				}
			}
			MPI_Recv(&M[p+k][k], 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}	
		if (k==n-1)
		{	//put 0's above diagonal
			for(int i=0; i<n-1; i++)
			{
				for(int j=i+1; j<n; j++)
				{
					M[i][j]=0;	
				}	
			}
			
			//print result to the screen
			printf("Cholesky decomposition of matrix M: L= \n");
			for (int i = 0; i <n; i++){
				for (int j = 0; j <n; j++){
					printf("%f", M[i][j]);
					if(j!=n-1)
					{
						printf(", ");	
					}
				}
				printf("\n");
			}
		}	
	}
	else if (world_rank<n-k){
		for (int c=0; c<=k; c++)//receving already calculated values
		{
			for (int r=c; r<n; r++)
			{
				MPI_Recv(&M[r][c], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
		
		for (int g=0; g<k; g++)//calculating non-diagonal elements concurrently
		{
			M[k+world_rank][k]-=M[k][g]*M[k+world_rank][g];		
		}	
		M[k+world_rank][k]/=M[k][k];
		MPI_Send(&M[k+world_rank][k], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}   
  }	  
  MPI_Finalize();
}  
