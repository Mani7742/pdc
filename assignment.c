#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define MAX_SIZE 2000  // Maximum array size (1D array)

// Function to initialize array
void initialize_array(double A[MAX_SIZE], int N) {
    srand(time(0));
    for (int i = 0; i < N; i++) {
        A[i] = rand() % 10000 + 1;
    }
}

// *Row-major equivalent for 1D array (just regular array traversal)*
void find_max_row_major(double A[MAX_SIZE], int N) {
    double total_time = 0.0;
    for (int r = 0; r < 10; r++) {
        double start = omp_get_wtime();
        double maxVal = A[0];

        #pragma omp parallel for schedule(static) reduction(max:maxVal)
        for (int i = 0; i < N; i++) {
            if (A[i] > maxVal) {
                maxVal = A[i];
            }
        }

        double end = omp_get_wtime();
        total_time += (end - start);
    }
    printf("Average(10) static Array Find Max Time: %.6f sec\n", total_time / 10);
}

void find_max_row_major1(double A[MAX_SIZE], int N) {
    double total_time = 0.0;
    for (int r = 0; r < 10; r++) {
        double start = omp_get_wtime();
        double maxVal = A[0];

        #pragma omp parallel for schedule(dynamic) reduction(max:maxVal)
        for (int i = 0; i < N; i++) {
            if (A[i] > maxVal) {
                maxVal = A[i];
            }
        }

        double end = omp_get_wtime();
        total_time += (end - start);
    }
    printf("Average(10) dynamic Array Find Max Time: %.6f sec\n", total_time / 10);
}
// *Column-major equivalent (not relevant for 1D array, same as row-major)*
// For a 1D array, there is no column/row distinction, so this is skipped or same as above.

int main() {
    int sizes[] = {512, 1024, 2000};  // Different array sizes
    int num_threads[] = {1, 4, 8};    // Different thread counts

    for (int t = 0; t < 3; t++) {
        omp_set_num_threads(num_threads[t]);
        printf("\n--- Testing with %d Threads ---\n", num_threads[t]);

        for (int s = 0; s < 3; s++) {
            int N = sizes[s];
            printf("\nArray Size: %d\n", N);

            static double A[MAX_SIZE];  // 1D array
            initialize_array(A, N);

            find_max_row_major(A, N);  // Only one way to traverse 1D array
            find_max_row_major1(A, N);  // Only one way to traverse 1D array
        }
    }

    return 0;
}