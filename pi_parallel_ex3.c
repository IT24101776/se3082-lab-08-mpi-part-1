#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char** argv) {
    int rank, size;
    long long total_iterations = 10000000;
    long long local_iterations;
    long long local_count = 0;
    long long global_count = 0;
    double x, y, pi_estimate;

    MPI_Init(&argc, &argv);
    double start_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate how many iterations this process will handle
    local_iterations = total_iterations / size;
    
    // Add any remaining iterations to the last process
    if (rank == size - 1) {
        local_iterations += total_iterations % size;
    }

    // Seed the random number generator. 
    unsigned int seed = time(NULL) + rank * 12345;
    srand(seed);

    // Monte Carlo Method calculation
    for (long long i = 0; i < local_iterations; i++) {
        // Generate random numbers between 0.0 and 1.0
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        
        // Check if the point falls inside the unit circle
        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }

    printf("Process %d: Performed %lld iterations, found %lld points inside.\n", rank, local_iterations, local_count);

    // Combine all local counts into the global count on rank 0
    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Estimate Pi: Area_circle / Area_square = (Pi * r^2) / (2r)^2 = Pi / 4
        // Pi = 4 * (Points_inside / Total_points)
        pi_estimate = 4.0 * (double)global_count / (double)total_iterations;
        
        printf("----------------------------------------------------\n");
        printf("Total points        = %lld\n", total_iterations);
        printf("Points inside circle= %lld\n", global_count);
        printf("Estimated Pi        = %f\n", pi_estimate);
        printf("----------------------------------------------------\n");
    }

    end_time = MPI_Wtime();
    if (rank == 0) printf("\nTIME_TAKEN: %f\n", end_time - start_time);
    MPI_Finalize();
    return 0;
}
