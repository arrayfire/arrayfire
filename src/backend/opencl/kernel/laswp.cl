/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       auto-converted from zlaswp.cu

       @author Stan Tomov
       @author Mathieu Faverge
       @author Ichitaro Yamazaki
       @author Mark Gates
*/

#define MAX_PIVOTS 32
typedef struct {
    int npivots;
    int ipiv[MAX_PIVOTS];
} zlaswp_params_t;

// Matrix A is stored row-wise in dAT.
// Divide matrix A into block-columns of NTHREADS columns each.
// Each GPU block processes one block-column of A.
// Each thread goes down a column of A,
// swapping rows according to pivots stored in params.
__kernel void laswp(int n, __global T *dAT, unsigned long dAT_offset,
                    int ldda, zlaswp_params_t params )
{
    dAT += dAT_offset;

    int tid = get_local_id(0) + get_local_size(0)*get_group_id(0);
    if ( tid < n ) {
        dAT += tid;
        __global T *A1  = dAT;

        for( int i1 = 0; i1 < params.npivots; ++i1 ) {
            int i2 = params.ipiv[i1];
            __global T *A2 = dAT + i2*ldda;
            T temp = *A1;
            *A1 = *A2;
            *A2 = temp;
            A1 += ldda;  // A1 = dA + i1*ldx
        }
    }
}
