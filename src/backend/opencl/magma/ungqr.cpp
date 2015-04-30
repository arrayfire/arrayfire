/*
    -- clMAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "magma.h"
#include "magma_blas.h"
#include "magma_data.h"
#include "magma_cpu_lapack.h"
#include "magma_helper.h"
#include "magma_sync.h"

template<typename Ty>  magma_int_t
magma_ungqr_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    cl_mem dA, size_t dA_offset, magma_int_t ldda,
    Ty *tau,
    cl_mem dT, size_t dT_offset, magma_int_t nb,
    magma_queue_t queue,
    magma_int_t *info)
{
#define dA(i,j) (dA),  ((i) + (j)*ldda)
#define dT(j)   (dT),  ((j)*nb)

    static const Ty c_zero = magma_zero<Ty>();
    static const Ty c_one  = magma_one<Ty>();

    magma_int_t m_kk, n_kk, k_kk, mi;
    magma_int_t lwork, lpanel;
    magma_int_t i, ib, ki, kk, iinfo;
    magma_int_t lddwork;
    cl_mem dV;
    Ty *work, *panel;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if ((n < 0) || (n > m)) {
        *info = -2;
    } else if ((k < 0) || (k > n)) {
        *info = -3;
    } else if (ldda < std::max(1,m)) {
        *info = -5;
    }
    if (*info != 0) {
        //magma_xerbla( __func__, -(*info));
        return *info;
    }

    if (n <= 0) {
        return *info;
    }

    // first kk columns are handled by blocked method.
    // ki is start of 2nd-to-last block
    if ((nb > 1) && (nb < k)) {
        ki = (k - nb - 1) / nb * nb;
        kk = std::min(k, ki+nb);
    } else {
        ki = 0;
        kk = 0;
    }

    // Allocate CPU work space
    // n*nb for zungqr workspace
    // (m - kk)*(n - kk) for last block's panel
    lwork = n*nb;
    lpanel = (m - kk)*(n - kk);
    magma_malloc_cpu<Ty>(&work, lwork + lpanel);
    if (work == NULL) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    panel = work + lwork;

    // Allocate work space on GPU
    if (MAGMA_SUCCESS != magma_malloc<Ty>(&dV, ldda*nb)) {
        magma_free_cpu(work);
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    // dT workspace has:
    // 2*std::min(m,n)*nb      for T and R^{-1} matrices from geqrf
    // ((n+31)/32*32)*nb for dW larfb workspace.
    lddwork = std::min(m,n);
    cl_mem dW;
    magma_malloc<Ty>(&dW, (((n+31)/32)*32)*nb);

    ungqr_work_func<Ty> cpu_ungqr;

    // Use unblocked code for the last or only block.
    if (kk < n) {
        m_kk = m - kk;
        n_kk = n - kk;
        k_kk = k - kk;
        magma_getmatrix<Ty>(m_kk, k_kk,
                            dA(kk, kk), ldda, panel, m_kk, queue);

        iinfo = cpu_ungqr(LAPACK_COL_MAJOR,
                          m_kk, n_kk, k_kk,
                          panel, m_kk,
                          &tau[kk], work, lwork);

        magma_setmatrix<Ty>(m_kk, n_kk,
                            panel, m_kk, dA(kk, kk), ldda, queue);

        // Set A(1:kk,kk+1:n) to zero.
        magmablas_laset<Ty>(MagmaFull, kk, n - kk, c_zero, c_zero, dA(0, kk), ldda, queue);
    }

    if (kk > 0) {
        // Use blocked code
        // stream:  copy Aii to V --> laset --> laset --> larfb --> [next]
        // CPU has no computation

        for (i = ki; i >= 0; i -= nb) {
            ib = std::min(nb, k-i);
            mi = m - i;

            // Copy current panel on the GPU from dA to dV
            magma_copymatrix<Ty>(mi, ib,
                                 dA(i,i), ldda,
                                 dV, 0,   ldda, queue);

            // set panel to identity
            magmablas_laset<Ty>(MagmaFull, i,  ib, c_zero, c_zero, dA(0, i), ldda, queue);
            magmablas_laset<Ty>(MagmaFull, mi, ib, c_zero, c_one,  dA(i, i), ldda, queue);

            if (i < n) {

                // Apply H to A(i:m,i:n) from the left
                magma_larfb_gpu<Ty>(MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise,
                                    mi, n-i, ib,
                                    dV, 0,    ldda, dT(i), nb,
                                    dA(i, i), ldda, dW, 0, lddwork, queue);
            }
        }
    }

    magma_free(dV);
    magma_free(dW);
    magma_free_cpu(work);
    return *info;

}

#define INSTANTIATE(T)                                                  \
    template  magma_int_t                                               \
    magma_ungqr_gpu<T>(magma_int_t m, magma_int_t n, magma_int_t k,     \
                       cl_mem dA, size_t dA_offset, magma_int_t ldda,   \
                       T *tau,                                          \
                       cl_mem dT, size_t dT_offset, magma_int_t nb,     \
                       magma_queue_t queue,                             \
                       magma_int_t *info);                              \

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(magmaFloatComplex)
INSTANTIATE(magmaDoubleComplex)
