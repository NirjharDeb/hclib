#include <math.h>
#include <shmem.h>
extern "C" {
#include "spmat.h"
}
#include <std_options.h>
#include "selector.h"

#define THREADS shmem_n_pes()
#define MYTHREAD shmem_my_pe()

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
    int64_t* cnt_;
    sparsemat_t* mat_;

    // Process each triangle check for lower triangular matrix
    void req_process(TrianglePkt pkg, int sender_rank) {
        int64_t tempCount = 0;

        // Loop over all neighbors of `vj` in the lower triangular matrix
        for (int64_t k = mat_->loffset[pkg.vj]; k < mat_->loffset[pkg.vj + 1]; k++) {
            if (pkg.w == mat_->lnonzero[k]) {
                tempCount++;
                break;
            }
            if (pkg.w < mat_->lnonzero[k]) {
                break;  // Since the neighbors are sorted, stop early
            }
        }

        // Update the global triangle count
        *cnt_ += tempCount;
    }
};

// Main triangle counting function for the lower triangular matrix
double triangle_selector(int64_t* count, sparsemat_t* L) {
    int64_t numpushed = 0;

    if (!L) {
        T0_printf("ERROR: triangle_selector: NULL L!\n");
        return -1.0;
    }

    // Start timing
    double t1 = wall_seconds();
  
    // Selector object for message passing
    TriangleSelector* triSelector = new TriangleSelector(count, L);

    hclib::finish([=, &numpushed]() {
        triSelector->start();
        int64_t k, kk, pe;
        int64_t l_i, L_i, L_j;

        TrianglePkt pkg;

        // Loop over nonzero elements in the lower triangular matrix `L`
        for (l_i = 0; l_i < L->lnumrows; l_i++) {
            // For each row i, loop over all its neighbors j (L_i -> L_j)
            for (k = L->loffset[l_i]; k < L->loffset[l_i + 1]; k++) {
                L_i = l_i * THREADS + MYTHREAD;
                L_j = L->lnonzero[k];

                // Determine which PE should handle this message
                pe = L_j % THREADS;
                pkg.vj = L_j / THREADS;  // Send vertex j

                // Loop over the neighbors of i and check for triangles
                for (kk = L->loffset[l_i]; kk < L->loffset[l_i + 1]; kk++) {
                    pkg.w = L->lnonzero[kk];  // Send vertex w

                    // Only count the triangle if w < j to avoid double-counting
                    if (pkg.w > L_j) {
                        continue;  // Skip redundant triangle checks
                    }

                    numpushed++;
                    triSelector->send(REQUEST, pkg, pe);
                }
            }
        }

        // Done sending all triangle counting messages
        triSelector->done(REQUEST);
    });

    // Wait for all PEs to finish
    lgp_barrier();

    t1 = wall_seconds() - t1;
    return t1;
}

// Function to validate triangle counts
void validate_triangle_counts(sparsemat_t* L, int64_t* triangles) {
    int64_t total_triangles = 0;
    int64_t total_triangle_counts = 0;

    // Sum the triangles across all PEs
    for (int64_t i = 0; i < L->lnumrows; i++) {
        total_triangle_counts += triangles[i];
    }

    // Reduce the triangle count across all PEs
    total_triangle_counts = lgp_reduce_add_l(total_triangle_counts);
    T0_printf("Total triangle counts: %ld\n", total_triangle_counts);

    // Ensure all reductions are finished across PEs
    lgp_barrier();
}

int main(int argc, char* argv[]) {
    const char *deps[] = { "system", "bale_actor" };
    hclib::launch(deps, 2, [=] {

        int64_t l_numrows = 10000;  
        double t1;

        int64_t* triangles = (int64_t*) malloc(l_numrows * sizeof(int64_t));
        if (!triangles) {
            T0_printf("ERROR: Memory allocation failed for triangles.\n");
            return -1;
        }
        memset(triangles, 0, l_numrows * sizeof(int64_t));

        sparsemat_t *L;

        int64_t numrows = l_numrows * THREADS;
        double erdos_renyi_prob = 0.1;
        L = erdos_renyi_random_graph(numrows, erdos_renyi_prob, UNDIRECTED, NOLOOPS, 12345);

        lgp_barrier();
        if (!L) {
            T0_printf("ERROR: Failed to generate matrix L!\n");
            free(triangles);
            return -1;
        }

        if (L->numrows <= 0 || L->lnumrows <= 0) {
            T0_printf("ERROR: Invalid matrix dimensions.\n");
            free(triangles);
            return -1;
        }

        T0_printf("L has %ld rows/cols and %ld nonzeros.\n", L->numrows, L->nnz);

        lgp_barrier();

        T0_printf("Run triangle counting ...\n");
        t1 = triangle_selector(triangles, L);
        T0_printf("Triangle counting completed in %8.3lf seconds.\n", t1);

        // Validate triangle counts
        validate_triangle_counts(L, triangles);

        free(triangles);

        return 0;
    });

    return 0;
}
