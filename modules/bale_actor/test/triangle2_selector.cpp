/******************************************************************
//
//
//  Modified code based on original by Institute for Defense Analyses
//  under BSD-like license.
//
//  This software is provided under the terms of the license at the top
//  of the original code.
//
 *****************************************************************/
/*! \file triangle_counting.cpp
 * \brief Triangle counting in a lower triangular graph.
 */

#include <math.h>
#include <shmem.h>
extern "C" {
#include "spmat.h"
}
#include <std_options.h>
#include "selector.h"

#define NUM_THREADS shmem_n_pes()
#define MY_PE shmem_my_pe()

typedef struct TriangleMessage {
    int64_t neighbor_vertex;
    int64_t current_vertex;
} TriangleMessage;

enum MailBox { REQUEST };

class TriangleCounter : public hclib::Selector<1, TriangleMessage> {
public:
    TriangleCounter(int64_t* triangle_count, sparsemat_t* lower_matrix)
        : triangle_count_(triangle_count), lower_matrix_(lower_matrix) {
        mb[REQUEST].process = [this](TriangleMessage msg, int sender_rank) {
            this->process_request(msg, sender_rank);
        };
    }

private:
    int64_t* triangle_count_;
    sparsemat_t* lower_matrix_;

    void process_request(TriangleMessage msg, int sender_rank) {
        int64_t triangle_found = 0;

        for (int64_t idx = lower_matrix_->loffset[msg.current_vertex];
             idx < lower_matrix_->loffset[msg.current_vertex + 1]; idx++) {
            if (msg.neighbor_vertex == lower_matrix_->lnonzero[idx]) {
                triangle_found++;
                break;
            }
            if (msg.neighbor_vertex < lower_matrix_->lnonzero[idx]) {
                break;
            }
        }

        // Atomically update the triangle count
        *triangle_count_ += triangle_found;
    }
};

double count_triangles(int64_t* triangle_count, int64_t* messages_sent, sparsemat_t* L) {
    int64_t local_messages = 0;

    if (!L) {
        T0_printf("ERROR: count_triangles: NULL matrix L!\n");
        assert(false);
    }

    // Start timing
    double start_time = wall_seconds();

    TriangleCounter* counter = new TriangleCounter(triangle_count, L);

    hclib::finish([=, &local_messages]() {
        counter->start();
        int64_t k, kk, dest_pe;
        int64_t local_row, global_row, neighbor_col;

        TriangleMessage msg;

        // Iterate over each nonzero element in the lower triangular matrix L
        for (local_row = 0; local_row < L->lnumrows; local_row++) {
            for (k = L->loffset[local_row]; k < L->loffset[local_row + 1]; k++) {
                global_row = local_row * NUM_THREADS + MY_PE;
                neighbor_col = L->lnonzero[k];

                dest_pe = neighbor_col % NUM_THREADS;
                msg.current_vertex = neighbor_col / NUM_THREADS;
                for (kk = L->loffset[local_row]; kk < L->loffset[local_row + 1]; kk++) {
                    msg.neighbor_vertex = L->lnonzero[kk];

                    if (msg.neighbor_vertex > neighbor_col) {
                        break;
                    }

                    local_messages++;
                    counter->send(REQUEST, msg, dest_pe);
                }
            }
        }
        // Indicate completion of message sending
        counter->done(REQUEST);
    });

    lgp_barrier();
    *messages_sent = local_messages;

    minavgmaxD_t stat[1];
    double elapsed_time = wall_seconds() - start_time;
    lgp_min_avg_max_d(stat, elapsed_time, NUM_THREADS);

    return elapsed_time;
}

int main(int argc, char* argv[]) {
    const char* deps[] = {"system", "bale_actor"};
    hclib::launch(deps, 2, [=] {
        int64_t buffer_count = 1024;
        int64_t num_rows_per_thread = 10000;
        int64_t nonzeros_per_row = 35;
        int64_t read_from_file = 0;
        char filename[64];

        double start_time;
        int64_t i;
        double erdos_renyi_prob = 0.0;

        int opt;
        while ((opt = getopt(argc, argv, "hb:n:f:e:")) != -1) {
            switch (opt) {
                case 'h':
                    // Print help message
                    break;
                case 'b':
                    sscanf(optarg, "%ld", &buffer_count);
                    break;
                case 'n':
                    sscanf(optarg, "%ld", &num_rows_per_thread);
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

        int64_t total_num_rows = num_rows_per_thread * NUM_THREADS;
        if (erdos_renyi_prob == 0.0) {
            erdos_renyi_prob = (2.0 * (nonzeros_per_row - 1)) / total_num_rows;
            if (erdos_renyi_prob > 1.0)
                erdos_renyi_prob = 1.0;
        } else {
            nonzeros_per_row = erdos_renyi_prob * total_num_rows;
        }

        T0_fprintf(stderr, "Running triangle counting on %d threads\n", NUM_THREADS);
        if (!read_from_file) {
            T0_fprintf(stderr, "Number of rows per thread (-n): %ld\n", num_rows_per_thread);
            T0_fprintf(stderr, "Erdos-Renyi probability (-e): %g\n", erdos_renyi_prob);
        }

        double correct_triangle_count = -1;

        sparsemat_t* L;

        if (read_from_file) {
            L = read_matrix_mm_to_dist(filename);
            if (!L)
                assert(false);

            T0_fprintf(stderr, "Reading file %s...\n", filename);
            T0_fprintf(stderr, "Matrix L has %ld rows/cols and %ld nonzeros.\n", L->numrows, L->nnz);

            if (!is_lower_triangular(L, 0)) {
                T0_fprintf(stderr, "ERROR: Input matrix is not lower triangular!\n");
                assert(false);
            }

            sort_nonzeros(L);

        } else {
            L = erdos_renyi_random_graph(total_num_rows, erdos_renyi_prob, UNDIRECTED, NOLOOPS, 12345);
        }

        lgp_barrier();

        T0_fprintf(stderr, "Matrix L has %ld rows/cols and %ld nonzeros.\n", L->numrows, L->nnz);

        if (!is_lower_triangular(L, 0)) {
            T0_fprintf(stderr, "ERROR: Matrix L is not lower triangular!\n");
            assert(false);
        }

        T0_fprintf(stderr, "Starting triangle counting...\n");
        int64_t local_triangle_count = 0;
        int64_t total_triangle_count = 0;
        int64_t local_messages_sent = 0;
        int64_t total_messages_sent = 0;

        // Run the triangle counting algorithm
        T0_fprintf(stderr, "Executing triangle counting algorithm...\n");
        double elapsed_time = count_triangles(&local_triangle_count, &local_messages_sent, L);
        lgp_barrier();

        total_triangle_count = lgp_reduce_add_l(local_triangle_count);
        total_messages_sent = lgp_reduce_add_l(local_messages_sent);
        T0_fprintf(stderr, "  %8.3lf seconds: %16ld triangles\n", elapsed_time, total_triangle_count);

        if ((correct_triangle_count >= 0) && (total_triangle_count != (int64_t)correct_triangle_count)) {
            T0_fprintf(stderr, "ERROR: Incorrect triangle count!\n");
        }

        if (correct_triangle_count == -1) {
            correct_triangle_count = total_triangle_count;
        }

        lgp_barrier();
    });

    return 0;
}
