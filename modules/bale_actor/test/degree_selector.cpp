#include <math.h>
#include <shmem.h>
#include <assert.h>
extern "C" {
#include "spmat.h"
}
#include <std_options.h>
#include "selector.h"

#define THREADS shmem_n_pes()
#define MYTHREAD shmem_my_pe()

typedef struct DegreePkt {
    int64_t node;
} DegreePkt;

enum MailBoxType {REQUEST};

class DegreeSelector: public hclib::Selector<1, DegreePkt> {
public:
    DegreeSelector(int64_t* degrees, sparsemat_t* mat) : degrees_(degrees), mat_(mat) {
        mb[REQUEST].process = [this](DegreePkt pkt, int sender_rank) { 
            this->req_process(pkt, sender_rank);
        };
    }

private:
    int64_t* degrees_;
    sparsemat_t* mat_;

    void req_process(DegreePkt pkg, int sender_rank) {
        if (pkg.node < 0 || pkg.node >= mat_->lnumrows) {
            T0_printf("ERROR: Invalid node index %ld\n", pkg.node);
            return;
        }

        int64_t degree_count = 0;
        for (int64_t k = mat_->loffset[pkg.node]; k < mat_->loffset[pkg.node + 1]; k++) {
            if (k >= mat_->lnnz) {
                T0_printf("ERROR: Invalid index in lnonzero %ld\n", k);
                return;
            }
            degree_count++;
        }

        degrees_[pkg.node] += degree_count;
    }
};

double degree_selector(int64_t* degrees, sparsemat_t* L) {
    if (!L) {
        T0_printf("ERROR: degree_selector: NULL L!\n");
        return -1.0;
    }

    double t1 = wall_seconds();

    DegreeSelector* degSelector = new DegreeSelector(degrees, L);

    hclib::finish([=]() {
        degSelector->start();
        int64_t l_i;

        DegreePkt pkg;
        for (l_i = 0; l_i < L->lnumrows; l_i++) {
            pkg.node = l_i;

            if (pkg.node < 0 || pkg.node >= L->lnumrows) {
                T0_printf("ERROR: Invalid node index %ld\n", pkg.node);
                continue;
            }

            int64_t pe = l_i % THREADS;
            degSelector->send(REQUEST, pkg, pe);
        }

        degSelector->done(REQUEST);
    });

    lgp_barrier();

    t1 = wall_seconds() - t1;
    return t1;
}

int main(int argc, char* argv[]) {
    const char *deps[] = { "system", "bale_actor" };
    hclib::launch(deps, 2, [=] {

        int64_t l_numrows = 10000;  
        double t1;

        int64_t* degrees = (int64_t*) malloc(l_numrows * sizeof(int64_t));
        if (!degrees) {
            T0_printf("ERROR: Memory allocation failed for degrees.\n");
            return -1;  // Explicit return in case of error
        }
        memset(degrees, 0, l_numrows * sizeof(int64_t));

        sparsemat_t *L;

        int64_t numrows = l_numrows * THREADS;
        double erdos_renyi_prob = 0.1;
        L = erdos_renyi_random_graph(numrows, erdos_renyi_prob, UNDIRECTED, NOLOOPS, 12345);

        lgp_barrier();
        if (!L) {
            T0_printf("ERROR: Failed to generate matrix L!\n");
            free(degrees);
            return -1;  // Explicit return in case of error
        }

        if (L->numrows <= 0 || L->lnumrows <= 0) {
            T0_printf("ERROR: Invalid matrix dimensions.\n");
            free(degrees);
            return -1;  // Explicit return in case of error
        }

        T0_printf("L has %ld rows/cols and %ld nonzeros.\n", L->numrows, L->nnz);

        lgp_barrier();

        T0_printf("Run degree counting ...\n");
        t1 = degree_selector(degrees, L);
        T0_printf("Degree counting completed in %8.3lf seconds.\n", t1);

        int64_t local_degree_count = 0;
        for (int64_t i = 0; i < l_numrows; ++i) {
            local_degree_count += degrees[i];
        }

        int64_t total_degree_counts = lgp_reduce_add_l(local_degree_count);
        T0_printf("Total degrees: %ld\n", total_degree_counts);

        free(degrees);

        return 0;
    });

    return 0;
}
