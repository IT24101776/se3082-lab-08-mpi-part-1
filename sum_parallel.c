#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    long long N = 1000000000; // 10 million
    
    MPI_Init(&argc, &argv);
    double start_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Calculate the chunk of numbers this process will handle
    long long chunk_size = N / size;
    long long start = rank * chunk_size + 1;
    
    // The last process handles any remaining numbers
    long long end = (rank == size - 1) ? N : start + chunk_size - 1;
    
    // Calculate local sum
    long long local_sum = 0;
    for (long long i = start; i <= end; ++i) {
        local_sum += i;
    }
    
    printf("Process %d: Summing %lld to %lld -> local sum = %lld\n", rank, start, end, local_sum);
    
    // Combine all local sums into the global sum on rank 0
    long long global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("----------------------------------------------------\n");
        printf("Total sum of 1 to %lld is %lld\n", N, global_sum);
        printf("----------------------------------------------------\n");
    }
    
    end_time = MPI_Wtime();
    if (rank == 0) printf("\nTIME_TAKEN: %f\n", end_time - start_time);
    MPI_Finalize();
    return 0;
}
