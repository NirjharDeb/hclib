/******************************************************************
//
//  Copyright(C) 2018, Institute for Defense Analyses
//  4850 Mark Center Drive, Alexandria, VA; 703-845-2500
//  This material may be reproduced by or for the US Government
//  pursuant to the copyright license under the clauses at DFARS
//  252.227-7013 and 252.227-7014.
//
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the
//      names of its contributors may be used to endorse or promote products
//      derived from this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
//  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
//  COPYRIGHT HOLDER NOR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
//  OF THE POSSIBILITY OF SUCH DAMAGE.
//
*****************************************************************/

/*! \file toposort.upc
 * \brief Demo application that does a toposort on a permuted upper triangular matrix
 */
#include "shmem.h"
extern "C" {
#include "spmat.h"
}
#include <std_options.h>
#include "selector.h"
#include <cstdio>
#include <cstdlib>

// --- Begin message counter support ---
// We allocate the per-thread counters dynamically so that THREADS need not be a compile-time constant.
long long * msg_sent;
long long * msg_recv;

void init_msg_counters() {
    msg_sent = new long long[THREADS];
    msg_recv = new long long[THREADS];
    for (int i = 0; i < THREADS; i++) {
         msg_sent[i] = 0;
         msg_recv[i] = 0;
    }
}

void report_msg_stats() {
    long long total_sent = 0, total_recv = 0;
    for (int i = 0; i < THREADS; i++) {
        total_sent += msg_sent[i];
        total_recv += msg_recv[i];
    }
    printf("Global messages sent: %lld, received: %lld\n", total_sent, total_recv);
}
// --- End message counter support ---

typedef struct pkg_topo_t {
    int64_t row;
    int64_t col;
    int64_t level;
} pkg_topo_t;

typedef struct pkg_cperm_t {
    int64_t pos;
    int64_t col;
} pkg_cperm_t;

class TopoSort: public hclib::Selector<1, pkg_topo_t> {
public:
    TopoSort(sparsemat_t *tmat, int64_t *lrowsum, int64_t *lrowcnt, int64_t *level, int64_t *matched_col, int64_t lnc, int64_t lnr)
      : tmat(tmat), lrowsum(lrowsum), lrowcnt(lrowcnt), level(level), matched_col(matched_col), lnc(lnc), lnr(lnr), pivot_count(0) {
        finalized = (bool*) calloc(lnr, sizeof(bool));
        for (int64_t i = 0; i < lnr; i++) {
            finalized[i] = false;
        }
        terminated = false;
        mb[0].process = [this](pkg_topo_t pkg, int sender_rank) { this->process0(pkg, sender_rank); };
    }
    int64_t getNumLevels() { return num_levels; }

private:
    sparsemat_t *tmat;
    int64_t *lrowsum;
    int64_t *lrowcnt;
    int64_t *level;
    int64_t *matched_col;
    int64_t num_levels = 0;
    uint64_t type_mask = 0x8000000000000000;
    int64_t lnc;
    int64_t lnr;
    int64_t pivot_count;
    bool *finalized;
    bool terminated;

public:
    void mark_finalized(int64_t row) {
        if (!finalized[row]) {
            finalized[row] = true;
            pivot_count++;
        }
        if (pivot_count >= lnr && !terminated) {
            terminated = true;
            initiate_global_done();
        }
    }

private:
    void process0(pkg_topo_t pkg_ptr, int sender_rank) {
        // Count a received message for this PE.
        msg_recv[MYTHREAD]++;
        if (pkg_ptr.row & type_mask) {
            int64_t curr_col = (pkg_ptr.col) / THREADS;
            int64_t col_level = pkg_ptr.level;
            pkg_topo_t pkg;
            for (int64_t colstart = tmat->loffset[curr_col]; colstart < tmat->loffset[curr_col+1]; colstart++) {
                int64_t row = tmat->lnonzero[colstart];
                pkg.row = row / THREADS;
                pkg.col = curr_col * THREADS + MYTHREAD;
                pkg.level = col_level;
                int64_t pe = row % THREADS;
                // Count a sent message.
                msg_sent[MYTHREAD]++;
                send(0, pkg, pe);
            }
        } else {
            if (finalized[pkg_ptr.row]) {
                return;
            }
            lrowsum[pkg_ptr.row] -= pkg_ptr.col;
            lrowcnt[pkg_ptr.row]--;
            if(pkg_ptr.level >= level[pkg_ptr.row]){
                level[pkg_ptr.row] = pkg_ptr.level + 1;
                if((pkg_ptr.level+1) > num_levels)
                    num_levels = pkg_ptr.level + 1;
            }
            if(lrowcnt[pkg_ptr.row] == 1 && !finalized[pkg_ptr.row]){
                int64_t row = pkg_ptr.row;
                mark_finalized(row);
                pkg_topo_t pkg;
                pkg.row = row | type_mask;
                pkg.col = lrowsum[row];
                pkg.level = level[row];
                matched_col[row] = pkg.col;
                int64_t pe = pkg.col % THREADS;
                // Count a sent message.
                msg_sent[MYTHREAD]++;
                send(0, pkg, pe);
            }
        }       
    }
};

class TopoSortCPerm: public hclib::Selector<1, pkg_cperm_t> {
    int64_t *lcperm;
    void process(pkg_cperm_t pkg, int sender_rank) {
        // Count a received message.
        msg_recv[MYTHREAD]++;
        lcperm[pkg.col/THREADS] = pkg.pos;
    }
public:
    TopoSortCPerm(int64_t *lcperm) : lcperm(lcperm) {
        mb[0].process = [this](pkg_cperm_t pkg, int sender_rank) { this->process(pkg, sender_rank); };
    }
};

double toposort_matrix_selector(SHARED int64_t *rperm, SHARED int64_t *cperm, sparsemat_t *mat, sparsemat_t *tmat) {
    int64_t nr = mat->numrows;
    int64_t nc = mat->numcols;
    int64_t lnr = (nr + THREADS - MYTHREAD - 1) / THREADS;
    int64_t lnc = (nc + THREADS - MYTHREAD - 1) / THREADS;

    int64_t * lrperm = lgp_local_part(int64_t, rperm);
    int64_t * lcperm = lgp_local_part(int64_t, cperm);

    uint64_t type_mask = 0x8000000000000000;
    int64_t * lrowqueue  = (int64_t*)calloc(lnr, sizeof(int64_t));
    int64_t * lcolqueue  = (int64_t*)calloc(lnc, sizeof(int64_t));
    int64_t * lcolqueue_level  = (int64_t*)calloc(lnc, sizeof(int64_t));
    int64_t * lrowsum    = (int64_t*)calloc(lnr, sizeof(int64_t));
    int64_t * lrowcnt    = (int64_t*)calloc(lnr, sizeof(int64_t));
    int64_t * level      = (int64_t*)calloc(lnr, sizeof(int64_t));
    int64_t * matched_col= (int64_t*)calloc(lnr, sizeof(int64_t));

    int64_t rownext, rowlast;
    int64_t colnext, collast;
    int64_t colstart, colend;
    rownext = rowlast = colnext = collast = colstart = colend = 0;

    for(int64_t i = 0; i < mat->lnumrows; i++){
        lrowsum[i] = 0L;
        lrowcnt[i] = mat->loffset[i+1] - mat->loffset[i];
        if(lrowcnt[i] == 1){
            lrowqueue[rowlast++] = i;
            level[i] = 0;
        }
        for(int64_t j = mat->loffset[i]; j < mat->loffset[i+1]; j++)
            lrowsum[i] += mat->lnonzero[j];
    }

    int64_t num_levels = 0;
    TopoSort *topo = new TopoSort(tmat, lrowsum, lrowcnt, level, matched_col, lnc, lnr);

    lgp_barrier();
    double t1 = wall_seconds();
    hclib::finish([=, &rowlast]() {
        topo->start();
        pkg_topo_t pkg;
        int64_t row, pe;
        for (int64_t i = 0; i < rowlast; i++) {
            row = pkg.row = lrowqueue[i];
            topo->mark_finalized(row);
            pkg.row |= type_mask;
            pkg.col = lrowsum[row];
            pkg.level = level[row];
            matched_col[row] = pkg.col;
            pe = pkg.col % THREADS;
            // Count a sent message.
            msg_sent[MYTHREAD]++;
            topo->send(0, pkg, pe);
        }
    });

    num_levels = topo->getNumLevels();
    delete topo;

    num_levels++;
    num_levels = lgp_reduce_max_l(num_levels);

    int64_t * level_sizes = (int64_t*)calloc(num_levels, sizeof(int64_t));
    int64_t * level_start = (int64_t*)calloc(num_levels, sizeof(int64_t));

    int64_t total = 0;
    for(int64_t i = 0; i < lnr; i++){
        level_sizes[level[i]]++;
    }

    for(int64_t i = 0; i < num_levels; i++){
        level_start[i] = total + lgp_prior_add_l(level_sizes[i]);
        level_sizes[i] = lgp_reduce_add_l(level_sizes[i]);
        total += level_sizes[i];
    }

    lgp_barrier();

    for(int64_t i = 0; i < lnr; i++){
        lrperm[i] = (nr - 1) - level_start[level[i]]++;
    }

    TopoSortCPerm *topocperm = new TopoSortCPerm(lcperm);
    hclib::finish([=]() {
        topocperm->start();
        for(int64_t i = 0; i < lnr; i++) {
            pkg_cperm_t pkg;
            pkg.pos = lrperm[i];
            pkg.col = matched_col[i];
            int64_t pe  = pkg.col % THREADS;
            // Count a sent message.
            msg_sent[MYTHREAD]++;
            topocperm->send(0, pkg, pe);
        }
        topocperm->done(0);
    });
    delete topocperm;
    lgp_barrier();

    minavgmaxD_t stat[1];
    t1 = wall_seconds() - t1;
    lgp_min_avg_max_d(stat, t1, THREADS);

    free(lrowcnt);
    free(lrowsum);
    free(lrowqueue);
    free(lcolqueue);

    return(stat->avg);
}

/*!
  \page toposort_page Toposort
  [Documentation omitted for brevity]
*/

static void usage(void) {
    T0_fprintf(stderr,"\
topo [-h][-b count][-M mask][-n num][-f filename][-Z num][-e prob][-D]\n\
 -h prints this help message\n\
 -b count is the number of packages in an exstack(2) buffer\n\
 -M mask is the or of 1,2,4,8,16 for the models: agi,exstack,exstack2,conveyor,alternate\n\
 -n num is the number of rows per thread\n\
 -f filename read the input matrix from filename (in Matrix Market format)\n\
 -Z num use an Erdos Renyi matrix with num being the expected number of nonzeros in a row \n\
 -e prob use an Erdos Renyi matrix where prob is the probability of an entry in matrix being non-zero \n\
 -D debugging flag that dumps out input and output files.\n\
\n");
    lgp_global_exit(0);
}

/*! \brief check the result toposort
 *
 * check that the permutations are in fact permutations and that applying
 * them to the original matrix yields an upper triangular matrix
 * \param mat the original matrix
 * \param rperminv the row permutation
 * \param cperminv the column permutation
 * \param dump_files debugging flag
 * \return 0 on success, 1 otherwise
 */
int check_is_triangle(sparsemat_t * mat, SHARED int64_t * rperminv, SHARED int64_t * cperminv, int64_t dump_files) {
    sparsemat_t * mat2;
    int ret = 0;

    int rf = is_perm(rperminv, mat->numrows);
    int cf = is_perm(cperminv, mat->numcols);
    if(!rf || !cf){
        T0_fprintf(stderr,"ERROR: check_is_triangle is_perm(rperminv2) = %d is_perm(cperminv2) = %d\n",rf,cf);
        return(1);
    }
    mat2 = permute_matrix(mat, rperminv, cperminv);
    if(!is_upper_triangular(mat2, 1)) {
        T0_fprintf(stderr,"ERROR: check_is_triangle fails\n");
        ret = 1;
    }
    clear_matrix(mat2);
    free(mat2);
    return(ret);
}

/*! \brief Generates an input matrix for the toposort algorithm
 * \param numrows the number of rows (and columns) in the produced matrix
 * \param prob the probability that there is an edge between two given vertices
 * \param rand_seed the seed for random number generation that determines the matrix and permutations
 * \return the permuted upper triangular matrix
 */
sparsemat_t * generate_toposort_input(int64_t numrows, double prob, int64_t rand_seed) {
    sparsemat_t * omat;
    int64_t numcols = numrows;

    T0_fprintf(stderr,"Creating input matrix for toposort\n"); fflush(stderr);
    double t = wall_seconds();
    omat = transpose_matrix(erdos_renyi_random_graph(numrows, prob, UNDIRECTED, LOOPS, rand_seed));
    T0_printf("generate ER graph time %lf\n", wall_seconds() - t);
    if(!omat) exit(1);
    if(!is_upper_triangular(omat, 1)) exit(1);

    t = wall_seconds();
    SHARED int64_t * rperminv = rand_permp(numrows, 1230+MYTHREAD);
    SHARED int64_t * cperminv = rand_permp(numcols, 45+MYTHREAD);
    T0_printf("generate perms time %lf\n", wall_seconds() - t);
    lgp_barrier();

    if(!rperminv || !cperminv){
        T0_printf("ERROR: topo_rand_permp returns NULL!\n"); fflush(0);
        return(NULL);
    }

    int64_t * lrperminv = lgp_local_part(int64_t, rperminv);
    int64_t * lcperminv = lgp_local_part(int64_t, cperminv);

    lgp_barrier();
    t = wall_seconds();
    sparsemat_t * mat = permute_matrix(omat, rperminv, cperminv);
    T0_printf("permute matrix time %lf\n", wall_seconds() - t);

    if(!mat) {
        T0_printf("ERROR: permute_matrix returned NULL"); fflush(0);
        return(NULL);
    }

    lgp_barrier();

    clear_matrix(omat);
    free(omat);
    lgp_all_free(rperminv);
    lgp_all_free(cperminv);

    return(mat);
}

int main(int argc, char * argv[]) {

    const char *deps[] = { "system", "bale_actor" };
    hclib::launch(deps, 2, [=] {

        // Initialize per-thread message counters.
        init_msg_counters();

        int64_t i, j, fromth, lnnz, start, end;
        int64_t pe, row, col, idx;
        double t1;

        int64_t l_numrows = 100000;
        double  nz_per_row = 10;
        int64_t buf_cnt = 1024;
        int64_t rand_seed =  MYTHREAD*MYTHREAD*10000 + 5;
        int64_t numrows, numcols;
        int64_t pos = 0;

        double erdos_renyi_prob = 0.0;
        int64_t models_mask = ALL_Models;
        int64_t printhelp = 0;
        int64_t read_graph = 0;
        char filename[64];
        int64_t dump_files = 0;
        int64_t cores_per_node = 1;

        int opt;
        while( (opt = getopt(argc, argv, "hb:c:M:n:f:Z:p:")) != -1 ) {
            switch(opt) {
            case 'h': printhelp = 1; break;
            case 'b': sscanf(optarg,"%ld" , &buf_cnt);  break;
            case 'c': sscanf(optarg,"%ld" ,&cores_per_node); break;
            case 'M': sscanf(optarg,"%ld" , &models_mask);  break;
            case 'n': sscanf(optarg,"%ld" , &l_numrows);  break;
            case 'f': read_graph = 1; sscanf(optarg,"%s", filename); break;
            case 'Z': sscanf(optarg,"%lf" , &nz_per_row);  break;
            case 'e': sscanf(optarg,"%lf" , &erdos_renyi_prob);  break;
            case 'D': dump_files = 1; break;
            default:  break;
            }
        }
        if(printhelp) usage();

        numrows = l_numrows * THREADS;
        numcols = numrows;
        if(erdos_renyi_prob == 0.0){
            erdos_renyi_prob = (2.0*nz_per_row)/(numrows - 1);
            if(erdos_renyi_prob > 1.0)
                erdos_renyi_prob = 1.0;
        } else {
            nz_per_row = erdos_renyi_prob * numrows;
        }

        T0_fprintf(stderr,"Running toposort on %d threads\n", THREADS);
        T0_fprintf(stderr,"buf_cnt (stack size)           (-b)   %ld\n", buf_cnt);
        T0_fprintf(stderr,"Number of rows per thread      (-n)   %ld\n", l_numrows);
        T0_fprintf(stderr,"Avg # of nonzeros per row      (-Z)   %2.2lf\n", nz_per_row);
        T0_fprintf(stderr,"Erdos-Renyi edge probability   (-e)   %lf\n", erdos_renyi_prob);
        T0_fprintf(stderr,"task mask (M) = %ld (should be 1,2,4,8,16 for agi, exstack, exstack2, conveyors, alternates\n", models_mask);

        sparsemat_t * mat = generate_toposort_input(numrows, erdos_renyi_prob, rand_seed);
        if(!mat){ T0_printf("ERROR: mat is NULL!\n"); exit(1); }

        T0_printf("Input matrix has %ld rows and %ld nonzeros\n", mat->numrows, mat->nnz);

        sparsemat_t * tmat = transpose_matrix(mat);
        if(!tmat){ T0_printf("ERROR: tmat is NULL!\n"); exit(1); }

        lgp_barrier();

        T0_fprintf(stderr,"Run toposort on mat (and tmat) ...\n");
        SHARED int64_t *rperminv2 = (int64_t*)lgp_all_alloc(numrows, sizeof(int64_t));
        SHARED int64_t *cperminv2 = (int64_t*)lgp_all_alloc(numcols, sizeof(int64_t));
        double gb_th  = (mat->numrows + mat->numcols*2 + mat->nnz*2)*8;

        double laptime = 0.0;
        T0_fprintf(stderr," Selector: \n");
        laptime = toposort_matrix_selector(rperminv2, cperminv2, mat, tmat);

        lgp_barrier();
        T0_fprintf(stderr,"  %8.3lf seconds\n", laptime);

        if( check_is_triangle(mat, rperminv2, cperminv2, dump_files) ) {
            printf("\nERROR: After toposort_matrix_upc: mat2 is not upper-triangular!\n");
        }

        // Print the message counters only once (from PE 0)
        if (MYTHREAD == 0)
            report_msg_stats();

        lgp_barrier();
        lgp_finalize();
    });

    return(0);
}
