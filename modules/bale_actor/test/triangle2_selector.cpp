/******************************************************************
//
//  Triangle counting code with binary search optimization and verification.
//
//  Based on triangle.upc code style.
//
 *****************************************************************/
/*! \file triangle_counting.cpp
 *  \brief Triangle counting in a lower triangular graph with binary search optimization and verification.
 */

#include <math.h>
#include <shmem.h>
extern "C" {
#include "spmat.h"
}
#include <std_options.h>
#include "selector.h"

#define THREADS shmem_n_pes()
#define MYTHREAD shmem_my_pe()

using namespace std;

//Printing to a new file (DEBUGGING)
#include <fstream>
#include <iostream>
#include <string>

// Delete a folder (if it exists) and recreate it
void resetFolder(string folder_name) {
  int folderRemoval = system(("rm -rf " + folder_name).c_str());
  if (folderRemoval) {
    printf("Failed to delete folder.\n");
  }
  int folderCreation = system(("mkdir " + folder_name).c_str());
  if (folderCreation) {
    printf("Failed to create folder.\n");
  }
}

/**
 * Print variable to 1 file across all PEs (used for performance metrics)
 * Persistent across different sbatch runs
 * File can only be cleared by manually deleting it
 * 
 * WARNING: This method is only valid if the variable you print out is the same for all PEs. 
 * One such example of a variable is "laptime", which is the time it took for the selector to run
 * successfully.
*/
void outVariableToNewFileGlobal(string name, double_t value) {
  int pe = MYTHREAD;

  //Track number of times this method has been called across all PEs
  static unsigned int call_count = 0;
  call_count++;

  //If PE is 0 and this is the first call to method, update variable file
  if (call_count == 1 && pe == 0) {
    string file_name = "triangle_selector_" + name + "[" + to_string(THREADS) + "]" +  ".txt";

    ofstream output_file(file_name, ios::app);

    if (output_file.is_open()) {
      string new_line = to_string(value);
      output_file << new_line << endl;
      output_file.close();
    } else {
      printf(("Failed to write " + name + " to output file.\n").c_str());
    }
  }
}

typedef struct TrianglePkt {
    int64_t w;
    int64_t vj;
} TrianglePkt;

enum MailBoxType {REQUEST};

class TriangleSelector: public hclib::Selector<1, TrianglePkt> {
public:
    TriangleSelector(int64_t* cnt, sparsemat_t* mat) : cnt_(cnt), mat_(mat) {
        mb[REQUEST].process = [this] (TrianglePkt pkt, int sender_rank) {
            this->req_process(pkt, sender_rank);
        };
    }

private:
    // Shared variables
    int64_t* cnt_;
    sparsemat_t* mat_;

    void req_process(TrianglePkt pkg, int sender_rank) {
        // Binary search optimization
        int64_t start = mat_->loffset[pkg.vj];
        int64_t end = mat_->loffset[pkg.vj + 1];
        int64_t* neighbors = mat_->lnonzero;
        int64_t left = start;
        int64_t right = end - 1;

        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            int64_t current_neighbor = neighbors[mid];

            if (current_neighbor == pkg.w) {
                // Found a triangle
                (*cnt_)++;
                break;
            } else if (current_neighbor < pkg.w) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
};


double triangle_selector(int64_t* count, int64_t* sr, sparsemat_t* L, sparsemat_t* U, int64_t alg) {
    int64_t numpushed = 0;

    if (!L) {
        T0_printf("ERROR: triangle_selector: NULL L!\n");
        return -1;
    }

    // Start timing
    double t1 = wall_seconds();

    if (alg == 0) {
        TriangleSelector* triSelector = new TriangleSelector(count, L);

        hclib::finish([=, &numpushed]() {
            triSelector->start();
            int64_t k, kk, pe;
            int64_t l_i, L_i, L_j;

            TrianglePkt pkg;
            // For each nonzero (i, j) in L
            for (l_i = 0; l_i < L->lnumrows; l_i++) {
                L_i = l_i * THREADS + MYTHREAD;
                for (k = L->loffset[l_i]; k < L->loffset[l_i + 1]; k++) {
                    L_j = L->lnonzero[k];

                    pe = L_j % THREADS;
                    pkg.vj = L_j / THREADS;
                    for (kk = L->loffset[l_i]; kk < k; kk++) {
                        pkg.w = L->lnonzero[kk];

                        numpushed++;
                        triSelector->send(REQUEST, pkg, pe);
                    }
                }
            }
            // Indicate that we are done with sending messages to the REQUEST mailbox
            triSelector->done(REQUEST);
        });
    } else {
        if (!U) {
            T0_printf("ERROR: triangle_selector: NULL U!\n");
            assert(false);
        }

        TriangleSelector* triSelector = new TriangleSelector(count, U);

        hclib::finish([=, &numpushed] () {
            triSelector->start();
            int64_t k, kk, pe;
            int64_t l_i, L_i, L_j;
            TrianglePkt pkg;

            // For each nonzero (i, j) in L
            for (l_i = 0; l_i < L->lnumrows; l_i++) {
                L_i = l_i * THREADS + MYTHREAD;
                for (k = L->loffset[l_i]; k < L->loffset[l_i + 1]; k++) {
                    L_j = L->lnonzero[k];

                    pe = L_j % THREADS;
                    pkg.vj = L_j / THREADS;

                    for (kk = U->loffset[l_i]; kk < U->loffset[l_i + 1]; kk++) {
                        pkg.w = U->lnonzero[kk];
                        numpushed++;
                        triSelector->send(REQUEST, pkg, pe);
                    }
                }
            }
            // Indicate that we are done with sending messages to the REQUEST mailbox
            triSelector->done(REQUEST);
        });
    }

    lgp_barrier();
    *sr = numpushed;
    // The triangle count is updated within the Selector
    double elapsed = wall_seconds() - t1;

    return elapsed;
}


int main(int argc, char* argv[]) {
    const char *deps[] = { "system", "bale_actor" };

    hclib::launch(deps, 2, [=] {

        int64_t l_numrows = 10000;         // Number of rows per thread
        int64_t nz_per_row = 35;           // Target number of nonzeros per row
        int64_t read_graph = 0L;           // Flag to indicate if graph should be read from a file
        char filename[64];
        double erdos_renyi_prob = 0.0;
        int64_t alg = 0;                   // Algorithm selection (0 or 1)
        double correct_answer = -1;        // For verification

        // Parse command-line arguments
        int opt;
        while ((opt = getopt(argc, argv, "hn:f:e:a:")) != -1) {
            switch (opt) {
                case 'h':
                    // Display help message
                    fprintf(stderr, "Usage: %s [-n rows_per_thread] [-e erdos_renyi_prob] [-f filename] [-a alg]\n", argv[0]);
                    exit(0);
                case 'n':
                    sscanf(optarg,"%ld", &l_numrows);
                    break;
                case 'f':
                    read_graph = 1; sscanf(optarg,"%s", filename); break;
                case 'e':
                    sscanf(optarg,"%lg", &erdos_renyi_prob); break;
                case 'a':
                    sscanf(optarg,"%ld", &alg); break;
                default:  break;
            }
        }

        int64_t numrows = l_numrows * THREADS;
        if (erdos_renyi_prob == 0.0) { // Use nz_per_row to get erdos_renyi_prob
            erdos_renyi_prob = (2.0 * (nz_per_row - 1)) / numrows;
            if (erdos_renyi_prob > 1.0) erdos_renyi_prob = 1.0;
        } else {                     // Use erdos_renyi_prob to get nz_per_row
            nz_per_row = erdos_renyi_prob * numrows;
        }

        sparsemat_t *A = NULL, *L = NULL, *U = NULL;

        if (read_graph) {
            // Read the graph from the provided file
            A = read_matrix_mm_to_dist(filename);
            if (!A) assert(false);

            T0_fprintf(stderr,"Reading file %s...\n", filename);
            T0_fprintf(stderr, "A has %ld rows/cols and %ld nonzeros.\n", A->numrows, A->nnz);

            // Ensure the matrix is lower triangular
            if (!is_lower_triangular(A, 0)) {
                T0_fprintf(stderr, "Assuming symmetric matrix... using lower-triangular portion...\n");
                tril(A, -1);
                L = A;
            } else {
                L = A;
            }

            sort_nonzeros(L);

        } else {
            // Generate an Erdős-Rényi random graph
            L = erdos_renyi_random_graph(numrows, erdos_renyi_prob, UNDIRECTED, NOLOOPS, 12345);
            tril(L, -1);  // Keep only lower triangular part
            sort_nonzeros(L);
        }

        lgp_barrier();

        if (alg == 1)
            U = transpose_matrix(L);

        lgp_barrier();

        T0_fprintf(stderr, "L has %ld rows/cols and %ld nonzeros.\n", L->numrows, L->nnz);

        if (!is_lower_triangular(L, 0)) {
            T0_fprintf(stderr,"ERROR: L is not lower triangular!\n");
            assert(false);
        }

        // Verification steps similar to the provided code
        // Calculate degrees and related metrics
        int64_t *cc = (int64_t*)lgp_all_alloc(L->numrows, sizeof(int64_t));
        int64_t *l_cc = lgp_local_part(int64_t, cc);
        for (int64_t i = 0; i < L->lnumrows; i++)
            l_cc[i] = 0;

        lgp_barrier();

        // Calculate column sums (degree counts)
        for (int64_t i = 0; i < L->lnnz; i++) {
            int64_t lindex = L->lnonzero[i] / THREADS;
            int64_t pe = L->lnonzero[i] % THREADS;
            lgp_fetch_and_inc(&cc[lindex], pe);
        }

        lgp_barrier();

        // Compute metrics for verification
        int64_t rtimesc_calc = 0;
        for (int64_t i = 0; i < L->lnumrows; i++) {
            int64_t deg = L->loffset[i + 1] - L->loffset[i];
            rtimesc_calc += deg * l_cc[i];
        }

        int64_t rchoose2_calc = 0;
        for (int64_t i = 0; i < L->lnumrows; i++) {
            int64_t deg = L->loffset[i + 1] - L->loffset[i];
            rchoose2_calc += deg * (deg - 1) / 2;
        }

        int64_t cchoose2_calc = 0;
        for (int64_t i = 0; i < L->lnumrows; i++) {
            int64_t deg = l_cc[i];
            cchoose2_calc += deg * (deg - 1) / 2;
        }

        int64_t pulls_calc = 0;
        int64_t pushes_calc = 0;
        if (alg == 0) {
            pulls_calc = lgp_reduce_add_l(rtimesc_calc);
            pushes_calc = lgp_reduce_add_l(rchoose2_calc);
        } else {
            pushes_calc = lgp_reduce_add_l(rtimesc_calc);
            pulls_calc = lgp_reduce_add_l(cchoose2_calc);
        }

        lgp_all_free(cc);

        T0_fprintf(stderr,"Calculated: Pulls = %ld\n            Pushes = %ld\n\n", pulls_calc, pushes_calc);

        // Run triangle counting
        T0_fprintf(stderr, "Running triangle counting...\n");
        int64_t tri_cnt = 0;           // Partial count of triangles on this thread
        int64_t total_tri_cnt = 0;     // Total number of triangles across all threads
        int64_t sh_refs = 0;           // Number of shared references or messages sent
        int64_t total_sh_refs = 0;

        // Run the triangle counting algorithm
        double laptime = triangle_selector(&tri_cnt, &sh_refs, L, U, alg);
        lgp_barrier();

        total_tri_cnt = lgp_reduce_add_l(tri_cnt);
        total_sh_refs = lgp_reduce_add_l(sh_refs);
        T0_fprintf(stderr, "  %8.3lf seconds: %16ld triangles\n", laptime, total_tri_cnt);
        T0_fprintf(stderr, "  %16ld messages sent\n", total_sh_refs);

        // add laptime to external file
        outVariableToNewFileGlobal("laptime", laptime);

        // Verification of the result
        if (correct_answer >= 0 && total_tri_cnt != (int64_t)correct_answer) {
            T0_fprintf(stderr, "ERROR: Wrong answer!\n");
        } else if (correct_answer == -1) {
            correct_answer = total_tri_cnt;
        }

        lgp_barrier();

        // Clean up
        clear_matrix(L);
        if (alg == 1)
            clear_matrix(U);

    });

    return 0;
}
