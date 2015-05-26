/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef __MAGMA_H
#define __MAGMA_H

#include "magma_common.h"

template<typename Ty>
magma_int_t magma_getrf_gpu(magma_int_t m, magma_int_t n,
                            cl_mem dA, size_t dA_offset, magma_int_t ldda,
                            magma_int_t *ipiv,
                            magma_queue_t queue,
                            magma_int_t *info);

template<typename Ty>
magma_int_t magma_potrf_gpu(magma_uplo_t   uplo, magma_int_t    n,
                            cl_mem dA, size_t dA_offset, magma_int_t ldda,
                            magma_queue_t queue,
                            magma_int_t*   info);

template<typename Ty> magma_int_t
magma_larfb_gpu(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    cl_mem dV   , size_t dV_offset,    magma_int_t lddv,
    cl_mem dT   , size_t dT_offset,    magma_int_t lddt,
    cl_mem dC   , size_t dC_offset,    magma_int_t lddc,
    cl_mem dwork, size_t dwork_offset, magma_int_t ldwork,
    magma_queue_t queue);

template<typename Ty> magma_int_t
magma_geqrf2_gpu(
    magma_int_t m, magma_int_t n,
    cl_mem dA, size_t dA_offset, magma_int_t ldda,
    Ty *tau,
    magma_queue_t* queue,
    magma_int_t *info);

template<typename Ty> magma_int_t
magma_geqrf3_gpu(
    magma_int_t m, magma_int_t n,
    cl_mem dA, size_t dA_offset,  magma_int_t ldda,
    Ty *tau, cl_mem dT, size_t dT_offset,
    magma_queue_t queue,
    magma_int_t *info);

template<typename Ty>  magma_int_t
magma_unmqr_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    cl_mem dA, size_t dA_offset, magma_int_t ldda,
    Ty *tau,
    cl_mem dC, size_t dC_offset, magma_int_t lddc,
    Ty *hwork, magma_int_t lwork,
    cl_mem dT, size_t dT_offset, magma_int_t nb,
    magma_queue_t queue,
    magma_int_t *info);

#if 0  // Needs to be enabled when unmqr2 is enabled
template<typename Ty> magma_int_t
magma_unmqr2_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    cl_mem dA, size_t dA_offset, magma_int_t ldda,
    Ty    *tau,
    cl_mem dC, size_t dC_offset, magma_int_t lddc,
    Ty    *wA, magma_int_t ldwa,
    magma_queue_t queue,
    magma_int_t *info);
#endif

template<typename Ty>  magma_int_t
magma_ungqr_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    cl_mem dA, size_t dA_offset, magma_int_t ldda,
    Ty *tau,
    cl_mem dT, size_t dT_offset, magma_int_t nb,
    magma_queue_t queue,
    magma_int_t *info);

template<typename Ty>  magma_int_t
magma_getrs_gpu(magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                cl_mem dA, size_t dA_offset, magma_int_t ldda,
                magma_int_t *ipiv,
                cl_mem dB, size_t dB_offset, magma_int_t lddb,
                magma_queue_t queue,
                magma_int_t *info);

#endif
