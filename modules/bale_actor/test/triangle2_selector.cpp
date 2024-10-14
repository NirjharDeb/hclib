/******************************************************************
//
//  Modified code based on original by Institute for Defense Analyses
//  under BSD-like license.
//
//  This software is provided under the terms of the license at the top
//  of the original code.
//
******************************************************************/
/*! \file triangle_counting.cpp
 *  \brief Triangle counting in a lower triangular graph.
 */

#include <cmath>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <shmem.h>

// Include necessary headers for sparse matrix operations and options
extern "C" {
    #include "spmat.h"
}
#include <std_options.h>
#include "selector.h"

// Define macros for the number of processing elements and the current processing element
#define NUM_PES shmem_n_pes()
#define MY_PE shmem_my_pe()

// Structure to represent a message containing vertices for triangle checking
typedef struct {
    int64_t neighbor_vertex;  // The neighbor vertex to check
    int64_t target_vertex;    // The target vertex to compare with neighbor_vertex
} TriangleMessage;

// Enum for mailbox types used in message passing
enum MailBoxType { REQUEST_MAILBOX };

// Class for counting triangles using message passing with hclib::Selector
class TriangleCounter : public hclib::Selector<1, TriangleMessage> {
public:
    // Constructor: Initializes the triangle counter with the shared triangle count and the graph matrix
    TriangleCounter(int64_t* global_triangle_count, sparsemat_t* graph_matrix)
        : triangle_count_(global_triangle_count), matrix_(graph_matrix) {
        // Define the processing function for the REQUEST_MAILBOX
        mb[REQUEST_MAILBOX].process = [this](TriangleMessage msg, int sender_pe) {
            this->process_request(msg, sender_pe);
        };
    }

private:
    int64_t* triangle_count_;     // Shared variable to keep track of the total triangle count
    sparsemat_t* matrix_;         // Pointer to the lower triangular graph matrix

    // Function to process incoming triangle checking requests
    void process_request(TriangleMessage msg, int sender_pe) {
        int64_t triangles_found = 0;

        // Get the range of non-zero entries for the target vertex
        int64_t start = matrix_->loffset[msg.target_vertex];
        int64_t end = matrix_->loffset[msg.target_vertex + 1];

        // Iterate over the non-zero entries to check for the existence of the neighbor vertex
        for (int64_t idx = start; idx < end; idx++) {
            int64_t current_neighbor = matrix_->lnonzero[idx];

            // If the neighbor vertex matches, increment the triangle count
            if (msg.neighbor_vertex == current_neighbor) {
                triangles_found++;
                break;  // Triangle found, exit the loop
            }

            // Since the entries are sorted, we can break early if neighbor_vertex is smaller
            if (msg.neighbor_vertex < current_neighbor) {
                break;
            }
        }

        // Atomically update the shared triangle count
        *triangle_count_ += triangles_found;
    }
};

// Function to count triangles in the graph using the TriangleCounter class
double count_triangles(int64_t* triangle_count, int64_t* messages_sent, sparsemat_t* L) {
    if (!L) {
        fprintf(stderr, "ERROR: count_triangles: The graph matrix L is NULL!\n");
        assert(false);
    }

    int64_t local_messages_sent = 0;  // Local count of messages sent

    // Start the timer
    double start_time = wall_seconds();

    // Create an instance of TriangleCounter
    TriangleCounter* counter = new TriangleCounter(triangle_count, L);

    // Start the hclib::Selector and the triangle counting algorithm
    hclib::finish([=, &local_messages_sent]() {
        counter->start();  // Start the message processing

        // Iterate over each local row in the lower triangular matrix L
        for (int64_t local_row = 0; local_row < L->lnumrows; local_row++) {
            int64_t global_row = local_row * NUM_PES + MY_PE;  // Calculate the global row index

            // Get the range of neighbors for the current row
            int64_t row_start = L->loffset[local_row];
            int64_t row_end = L->loffset[local_row + 1];

            // Iterate over each neighbor in the current row
            for (int64_t idx = row_start; idx < row_end; idx++) {
                int64_t neighbor_col = L->lnonzero[idx];  // Neighbor column (vertex)

                // Determine the processing element (PE) responsible for neighbor_col
                int64_t dest_pe = neighbor_col % NUM_PES;
                int64_t dest_local_index = neighbor_col / NUM_PES;

                TriangleMessage msg;
                msg.target_vertex = dest_local_index;  // Set the target vertex in the message

                // Iterate again over the neighbors to create pairs for triangle checking
                for (int64_t idx2 = row_start; idx2 < row_end; idx2++) {
                    msg.neighbor_vertex = L->lnonzero[idx2];  // Neighbor vertex to check

                    // Since the matrix is lower triangular and sorted, break early if necessary
                    if (msg.neighbor_vertex > neighbor_col) {
                        break;
                    }

                    // Send the triangle checking request to the appropriate PE
                    local_messages_sent++;
                    counter->send(REQUEST_MAILBOX, msg, dest_pe);
                }
            }
        }

        // Indicate that no more messages will be sent to REQUEST_MAILBOX
        counter->done(REQUEST_MAILBOX);
    });

    // Synchronize all PEs
    lgp_barrier();

    *messages_sent = local_messages_sent;  // Update the total messages sent

    // Calculate the elapsed time
    double elapsed_time = wall_seconds() - start_time;

    return elapsed_time;
}

int main(int argc, char* argv[]) {
    // Dependencies for the hclib launcher
    const char* deps[] = { "system", "bale_actor" };
    hclib::launch(deps, 2, [=] {
        // Default parameters
        int64_t rows_per_pe = 10000;          // Number of rows per processing element
        int64_t nonzeros_per_row = 35;        // Average number of non-zeros per row
        int64_t read_from_file = 0;           // Flag to indicate if graph should be read from a file
        char filename[64];                    // Filename for the input graph
        double erdos_renyi_prob = 0.0;        // Probability for the Erdős-Rényi graph

        // Parse command-line arguments
        int opt;
        while ((opt = getopt(argc, argv, "hn:f:e:")) != -1) {
            switch (opt) {
                case 'h':
                    // Display help message
                    fprintf(stderr, "Usage: %s [-n rows_per_pe] [-e erdos_renyi_prob] [-f filename]\n", argv[0]);
                    exit(0);
                case 'n':
                    sscanf(optarg, "%ld", &rows_per_pe);
                    break;
                case 'f':
                    read_from_file = 1;
                    sscanf(optarg, "%s", filename);
                    break;
                case 'e':
                    sscanf(optarg, "%lg", &erdos_renyi_prob);
                    break;
                default:
                    break;
            }
        }

        // Calculate total number of rows in the graph
        int64_t total_rows = rows_per_pe * NUM_PES;

        // If probability is not set, calculate it based on the desired non-zeros per row
        if (erdos_renyi_prob == 0.0) {
            erdos_renyi_prob = (2.0 * (nonzeros_per_row - 1)) / total_rows;
            if (erdos_renyi_prob > 1.0)
                erdos_renyi_prob = 1.0;
        } else {
            nonzeros_per_row = erdos_renyi_prob * total_rows;
        }

        // Initialize the graph matrix
        sparsemat_t* L;

        if (read_from_file) {
            // Read the graph from the provided file
            L = read_matrix_mm_to_dist(filename);
            if (!L) {
                fprintf(stderr, "ERROR: Unable to read the graph from file %s\n", filename);
                assert(false);
            }

            // Check if the graph is lower triangular
            if (!is_lower_triangular(L, 0)) {
                fprintf(stderr, "ERROR: The input graph is not lower triangular.\n");
                assert(false);
            }

            // Ensure that the non-zero entries are sorted
            sort_nonzeros(L);
        } else {
            // Generate an Erdős-Rényi random graph
            L = erdos_renyi_random_graph(total_rows, erdos_renyi_prob, UNDIRECTED, NOLOOPS, 12345);

            // Ensure that the graph is lower triangular
            if (!is_lower_triangular(L, 0)) {
                // Convert to lower triangular form
                tril(L, -1);  // Keep only the lower triangular part
            }

            // Sort the non-zero entries
            sort_nonzeros(L);
        }

        // Synchronize all PEs
        lgp_barrier();

        // Verify that the matrix is lower triangular
        if (!is_lower_triangular(L, 0)) {
            fprintf(stderr, "ERROR: The matrix L is not lower triangular.\n");
            assert(false);
        }

        // Variables to hold triangle counts and messages sent
        int64_t local_triangle_count = 0;
        int64_t total_triangle_count = 0;
        int64_t local_messages_sent = 0;
        int64_t total_messages_sent = 0;

        // Run the triangle counting algorithm
        double computation_time = count_triangles(&local_triangle_count, &local_messages_sent, L);

        // Synchronize all PEs
        lgp_barrier();

        // Reduce the triangle counts and messages sent across all PEs
        total_triangle_count = lgp_reduce_add_l(local_triangle_count);
        total_messages_sent = lgp_reduce_add_l(local_messages_sent);

        // Display the final results
        T0_fprintf(stderr, "Triangle counting completed.\n");
        T0_fprintf(stderr, "Elapsed time: %8.3lf seconds\n", computation_time);
        T0_fprintf(stderr, "Total triangles found: %16ld\n", total_triangle_count);

        // Synchronize before exiting
        lgp_barrier();
    });

    return 0;
}
