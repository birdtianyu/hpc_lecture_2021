#include <mpi.h>
#include <omp.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <immintrin.h>
using namespace std;

int main(int argc, char** argv) {
  int size, rank;
  MPI_Init(&argc, &argv);                 // Initialization
  MPI_Comm_size(MPI_COMM_WORLD, &size);   // Get the number of processes in the process group
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Get the current process number

  const int N = 1024;
  int block_size = N/size; // block size
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N, 0);

  vector<float> subA(N*block_size);
  vector<float> subB(N*block_size);
  vector<float> subC(N*block_size, 0);
// A: Randomly initialize matrix A and matrix B
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
    }
  }
// B: Initialize matrix subA and matrix subB
  int offset = block_size*rank;
#pragma omp parallel for
  for (int i=0; i<block_size; i++)
    for (int j=0; j<N; j++)
      subA[N*i+j] = A[N*(i+offset)+j];
#pragma omp parallel for
  for (int i=0; i<N; i++)
    for (int j=0; j<block_size; j++)
      subB[block_size*i+j] = B[N*i+j+offset];


  int send_to = (rank - 1 + size) % size;
  int recv_from = (rank + 1) % size;

  double comp_time = 0, comm_time = 0;
  for(int irank=0; irank<size; irank++) {
    auto tic = chrono::steady_clock::now();  // start time
    offset = block_size*((rank+irank) % size);
// C: subC = subA * subB
#pragma omp parallel for
    for (int i=0; i<block_size; i++)
      for (int k=0; k<N; k++)
        for (int j=0; j<block_size; j++)
        {
          subC[N*i+j+offset] += subA[N*i+k] * subB[block_size*k+j];
        }
    auto toc = chrono::steady_clock::now();  // finish time
    comp_time += chrono::duration<double>(toc - tic).count(); // add time
    //    send: (buffer address, data size, data type, destination, tag，communicator)
    MPI_Send(&subB[0], N*block_size, MPI_FLOAT, send_to, 0, MPI_COMM_WORLD);
    // receive: (buffer address, data size, data type, source, tag，communicator, status)
    MPI_Recv(&subB[0], N*block_size, MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    tic = chrono::steady_clock::now();       // communication end time
    comm_time += chrono::duration<double>(tic - toc).count();  // add communication time
  }

  MPI_Allgather(&subC[0], N*block_size, MPI_FLOAT, &C[0], N*block_size, MPI_FLOAT, MPI_COMM_WORLD);


#pragma omp parallel for
  for (int i=0; i<N; i++)
    for (int k=0; k<N; k++)
      for (int j=0; j<N; j++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];

  // Count all errors
  double err = 0;
#pragma omp parallel for reduction(+:err)
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]); // fabs(x): Absolute value of x

  if(rank==0) {
    double time = comp_time+comm_time;
    printf("N    : %d\n",N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
  }
  MPI_Finalize();   // Exit MPI environment
}

