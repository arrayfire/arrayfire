/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/***********************************************************************
 * Based on MAGMA library http://icl.cs.utk.edu/magma/
 * Below is the original copyright.
 *
 *   -- MAGMA (version 1.1) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date
 *
 *      @author Mark Gates
 *      @precisions normal z -> s d c
 *
 * -- Innovative Computing Laboratory
 * -- Electrical Engineering and Computer Science Department
 * -- University of Tennessee
 * -- (C) Copyright 2009-2013
 *
 * Redistribution  and  use  in  source and binary forms, with or without
 * modification,  are  permitted  provided  that the following conditions
 * are met:
 *
 * * Redistributions  of  source  code  must  retain  the above copyright
 *   notice,  this  list  of  conditions  and  the  following  disclaimer.
 * * Redistributions  in  binary  form must reproduce the above copyright
 *   notice,  this list of conditions and the following disclaimer in the
 *   documentation  and/or other materials provided with the distribution.
 * * Neither  the  name of the University of Tennessee, Knoxville nor the
 *   names of its contributors may be used to endorse or promote products
 *   derived from this software without specific prior written permission.
 *
 * THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **********************************************************************/


#ifndef MAGMA_DATA_H
#define MAGMA_DATA_H
#include <iostream>

#include <platform.hpp>
#include "magma_types.h"

#define check_error( err ) if (err != CL_SUCCESS) throw cl::Error(err);

// ========================================
// memory allocation
// Allocate size bytes on GPU, returning pointer in ptrPtr.
template<typename T> static magma_int_t
magma_malloc( magma_ptr* ptrPtr, int num)
{
    size_t size = num * sizeof(T);
    // malloc and free sometimes don't work for size=0, so allocate some minimal size
    if ( size == 0 )
        size = sizeof(T);
    cl_int err;
    *ptrPtr = clCreateBuffer(opencl::getContext()(), CL_MEM_READ_WRITE, size, NULL, &err );
    if ( err != clblasSuccess ) {
        return MAGMA_ERR_DEVICE_ALLOC;
    }
    return MAGMA_SUCCESS;
}

// --------------------
// Free GPU memory allocated by magma_malloc.
static inline magma_int_t
magma_free(cl_mem ptr)
{
    cl_int err = clReleaseMemObject( ptr );
    if ( err != clblasSuccess ) {
        return MAGMA_ERR_INVALID_PTR;
    }
    return MAGMA_SUCCESS;
}

// --------------------
// Allocate size bytes on CPU, returning pointer in ptrPtr.
// The purpose of using this instead of malloc() is to properly align arrays
// for vector (SSE) instructions. The default implementation uses
// posix_memalign (on Linux, MacOS, etc.) or _aligned_malloc (on Windows)
// to align memory to a 32 byte boundary.
// Use magma_free_cpu() to free this memory.

template<typename T> static magma_int_t
magma_malloc_cpu(T** ptrPtr, int num)
{
    size_t size = num * sizeof(T);
    // malloc and free sometimes don't work for size=0, so allocate some minimal size
    if ( size == 0 )
        size = sizeof(T);
#if 1
    #if defined( _WIN32 ) || defined( _WIN64 )
    *ptrPtr = (T *)_aligned_malloc( size, 32 );
    if ( *ptrPtr == NULL ) {
        return MAGMA_ERR_HOST_ALLOC;
    }
    #else
    int err = posix_memalign((void **)ptrPtr, 32, size );
    if ( err != 0 ) {
        *ptrPtr = NULL;
        return MAGMA_ERR_HOST_ALLOC;
    }
    #endif
#else
    *ptrPtr = malloc( size );
    if ( *ptrPtr == NULL ) {
        return MAGMA_ERR_HOST_ALLOC;
    }
#endif
    return MAGMA_SUCCESS;
}

// --------------------
// Free CPU pinned memory previously allocated by magma_malloc_pinned.
// The default implementation uses free(), which works for both malloc and posix_memalign.
// For Windows, _aligned_free() is used.
template<typename T> static magma_int_t
magma_free_cpu(T* ptr )
{
#if defined( _WIN32 ) || defined( _WIN64 )
    _aligned_free( ptr );
#else
    free( ptr );
#endif
    return MAGMA_SUCCESS;
}

// ========================================
// copying vectors
template<typename T> static void
magma_setvector(
    magma_int_t n,
    T const* hx_src,                   magma_int_t incx,
    cl_mem   dy_dst, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if (n <= 0)
        return;

    if (incx == 1 && incy == 1) {
        cl_int err = clEnqueueWriteBuffer(
            queue, dy_dst, CL_TRUE,
            dy_offset*sizeof(T), n*sizeof(T),
            hx_src, 0, NULL, NULL);
        check_error( err );
    }
    else {
        magma_int_t ldha = incx;
        magma_int_t lddb = incy;
        magma_setmatrix( 1, n,
            hx_src,            ldha,
            dy_dst, dy_offset, lddb,
            queue);
    }
}

// --------------------
template<typename T> static void
magma_setvector_async(
    magma_int_t n,
    T const* hx_src,                   magma_int_t incx,
    cl_mem   dy_dst, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue, magma_event_t *event )
{
    if (n <= 0)
        return;

    if (incx == 1 && incy == 1) {
        cl_int err = clEnqueueWriteBuffer(
            queue, dy_dst, CL_FALSE,
            dy_offset*sizeof(T), n*sizeof(T),
            hx_src, 0, NULL, event);
        check_error( err );
    }
    else {
        magma_int_t ldha = incx;
        magma_int_t lddb = incy;
        magma_setmatrix_async( 1, n,
            hx_src,            ldha,
            dy_dst, dy_offset, lddb,
            queue, event);
    }
}

// --------------------
template<typename T> static void
magma_getvector(
    magma_int_t n,
    cl_mem dx_src, size_t dx_offset, magma_int_t incx,
    T*     hy_dst,                   magma_int_t incy,
    magma_queue_t queue )
{
    if (n <= 0)
        return;

    if (incx == 1 && incy == 1) {
        cl_int err = clEnqueueReadBuffer(
            queue, dx_src, CL_TRUE,
            dx_offset*sizeof(T), n*sizeof(T),
            hy_dst, 0, NULL, NULL);
        check_error( err );
    }
    else {
        magma_int_t ldda = incx;
        magma_int_t ldhb = incy;
        magma_getmatrix( 1, n,
            dx_src, dx_offset, ldda,
            hy_dst,            ldhb,
            queue);
    }
}

// --------------------
template<typename T> static void
magma_getvector_async(
    magma_int_t n,
    cl_mem dx_src, size_t dx_offset, magma_int_t incx,
    T*     hy_dst,                   magma_int_t incy,
    magma_queue_t queue, magma_event_t *event )
{
    if (n <= 0)
        return;

    if (incx == 1 && incy == 1) {
        cl_int err = clEnqueueReadBuffer(
            queue, dx_src, CL_FALSE,
            dx_offset*sizeof(T), n*sizeof(T),
            hy_dst, 0, NULL, event);
        check_error( err );
    }
    else {
        magma_int_t ldda = incx;
        magma_int_t ldhb = incy;
        magma_getmatrix_async( 1, n,
            dx_src, dx_offset, ldda,
            hy_dst,            ldhb,
            queue, event);
    }
}

// --------------------
template<typename T> static void
magma_copymatrix(
    magma_int_t m, magma_int_t n,
    cl_mem dA_src, size_t dA_offset, magma_int_t ldda,
    cl_mem dB_dst, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue )
{
    if (m <= 0 || n <= 0)
        return;

    size_t src_origin[3] = { dA_offset*sizeof(T), 0, 0 };
    size_t dst_orig[3]   = { dB_offset*sizeof(T), 0, 0 };
    size_t region[3]     = { m*sizeof(T), static_cast<size_t>(n), 1 };
    cl_int err = clEnqueueCopyBufferRect(
        queue, dA_src, dB_dst,
        src_origin, dst_orig, region,
        ldda*sizeof(T), 0,
        lddb*sizeof(T), 0,
        0, NULL, NULL );
    check_error( err );
}

// --------------------
template<typename T> static void
magma_copymatrix_async(
    magma_int_t m, magma_int_t n,
    cl_mem dA_src, size_t dA_offset, magma_int_t ldda,
    cl_mem dB_dst, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue, magma_event_t *event )
{
    if (m <= 0 || n <= 0)
        return;

    // TODO how to make non-blocking?
    size_t src_origin[3] = { dA_offset*sizeof(T), 0, 0 };
    size_t dst_orig[3]   = { dB_offset*sizeof(T), 0, 0 };
    size_t region[3]     = { m*sizeof(T), static_cast<size_t>(n), 1 };
    cl_int err = clEnqueueCopyBufferRect(
        queue, dA_src, dB_dst,
        src_origin, dst_orig, region,
        ldda*sizeof(T), 0,
        lddb*sizeof(T), 0,
        0, NULL, event );
    check_error( err );
}

// --------------------
template<typename T> static void
magma_copyvector(
    magma_int_t n,
    cl_mem dx_src, size_t dx_offset, magma_int_t incx,
    cl_mem dy_dst, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue )
{
    if (n <= 0)
        return;

    if (incx == 1 && incy == 1) {
        cl_int err = clEnqueueReadBuffer(
            queue, dx_src, CL_TRUE,
            dx_offset*sizeof(T), n*sizeof(T),
            dy_dst, dy_offset*sizeof(T), NULL, NULL);
        check_error( err );
    }
    else {
        magma_int_t ldda = incx;
        magma_int_t lddb = incy;
        magma_copymatrix<T>( 1, n,
            dx_src, dx_offset, ldda,
            dy_dst, dy_offset, lddb,
            queue);
    }
}

// --------------------
template<typename T> static void
magma_copyvector_async(
    magma_int_t n,
    cl_mem dx_src, size_t dx_offset, magma_int_t incx,
    cl_mem dy_dst, size_t dy_offset, magma_int_t incy,
    magma_queue_t queue, magma_event_t *event )
{
    if (n <= 0)
        return;

    if (incx == 1 && incy == 1) {
        cl_int err = clEnqueueReadBuffer(
            queue, dx_src, CL_FALSE,
            dx_offset*sizeof(T), n*sizeof(T),
            dy_dst, dy_offset*sizeof(T), NULL, event);
        check_error( err );
    }
    else {
        magma_int_t ldda = incx;
        magma_int_t lddb = incy;
        magma_copymatrix_async<T>( 1, n,
            dx_src, dx_offset, ldda,
            dy_dst, dy_offset, lddb,
            queue, event);
    }
}


// ========================================
// copying sub-matrices (contiguous columns)
// OpenCL takes queue even for blocking transfers, oddly.
template<typename T> static void
magma_setmatrix(
    magma_int_t m, magma_int_t n,
    T const* hA_src,                   magma_int_t ldha,
    cl_mem   dB_dst, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue )
{
    if (m <= 0 || n <= 0)
        return;

    size_t buffer_origin[3] = { dB_offset*sizeof(T), 0, 0 };
    size_t host_orig[3]     = { 0, 0, 0 };
    size_t region[3]        = { m*sizeof(T), (size_t)n, 1 };
    cl_int err = clEnqueueWriteBufferRect(
        queue, dB_dst, CL_TRUE,  // blocking
        buffer_origin, host_orig, region,
        lddb*sizeof(T), 0,
        ldha*sizeof(T), 0,
        hA_src, 0, NULL, NULL );
    check_error( err );
}

// --------------------
template<typename T> static void
magma_setmatrix_async(
    magma_int_t m, magma_int_t n,
    T const* hA_src,                   magma_int_t ldha,
    cl_mem   dB_dst, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue, magma_event_t *event )
{
    if (m <= 0 || n <= 0)
        return;

    size_t buffer_origin[3] = { dB_offset*sizeof(T), 0, 0 };
    size_t host_orig[3]     = { 0, 0, 0 };
    size_t region[3]        = { m*sizeof(T), (size_t)n, 1 };
    cl_int err = clEnqueueWriteBufferRect(
        queue, dB_dst, CL_FALSE,  // non-blocking
        buffer_origin, host_orig, region,
        lddb*sizeof(T), 0,
        ldha*sizeof(T), 0,
        hA_src, 0, NULL, event );
    clFlush(queue);
    check_error( err );
}

// --------------------
template<typename T> static void
magma_getmatrix(
    magma_int_t m, magma_int_t n,
    cl_mem dA_src, size_t dA_offset, magma_int_t ldda,
    T*     hB_dst,                   magma_int_t ldhb,
    magma_queue_t queue )
{
    if (m <= 0 || n <= 0)
       return;

    size_t buffer_origin[3] = { dA_offset*sizeof(T), 0, 0 };
    size_t host_orig[3]     = { 0, 0, 0 };
    size_t region[3]        = { m*sizeof(T), (size_t)n, 1 };
    cl_int err = clEnqueueReadBufferRect(
        queue, dA_src, CL_TRUE,  // blocking
        buffer_origin, host_orig, region,
        ldda*sizeof(T), 0,
        ldhb*sizeof(T), 0,
        hB_dst, 0, NULL, NULL );
    check_error( err );
}

// --------------------
template<typename T> static void
magma_getmatrix_async(
    magma_int_t m, magma_int_t n,
    cl_mem dA_src, size_t dA_offset, magma_int_t ldda,
    T*     hB_dst,                   magma_int_t ldhb,
    magma_queue_t queue, magma_event_t *event )
{
    if (m <= 0 || n <= 0)
        return;

    size_t buffer_origin[3] = { dA_offset*sizeof(T), 0, 0 };
    size_t host_orig[3]     = { 0, 0, 0 };
    size_t region[3]        = { m*sizeof(T), (size_t)n, 1 };
    cl_int err = clEnqueueReadBufferRect(
        queue, dA_src, CL_FALSE,  // non-blocking
        buffer_origin, host_orig, region,
        ldda*sizeof(T), 0,
        ldhb*sizeof(T), 0,
        hB_dst, 0, NULL, event );
    clFlush(queue);
    check_error( err );
}

template<typename T> void
magmablas_transpose_inplace(
    magma_int_t n,
    cl_mem dA, size_t dA_offset, magma_int_t ldda,
    magma_queue_t queue);

template<typename T> void
magmablas_transpose(
    magma_int_t m, magma_int_t n,
    cl_mem dA,  size_t dA_offset,  magma_int_t ldda,
    cl_mem dAT, size_t dAT_offset, magma_int_t lddat,
    magma_queue_t queue);

template<typename T> void
magmablas_laswp(
    magma_int_t n,
    cl_mem dAT, size_t dAT_offset, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue);

template<typename T> void
magmablas_swapdblk(magma_int_t n, magma_int_t nb,
                   cl_mem dA, magma_int_t dA_offset, magma_int_t ldda, magma_int_t inca,
                   cl_mem dB, magma_int_t dB_offset, magma_int_t lddb, magma_int_t incb,
                   magma_queue_t queue);

template<typename T> void
magmablas_laset(magma_uplo_t uplo, magma_int_t m, magma_int_t n,
                T offdiag, T diag,
                cl_mem dA, size_t dA_offset, magma_int_t ldda,
                magma_queue_t queue);

#if 0  // Needs to be enabled when unmqr2 is enabled
template<typename T> void
magmablas_laset_band(magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
                     T offdiag, T diag,
                     cl_mem dA, size_t dA_offset, magma_int_t ldda,
                     magma_queue_t queue);
#endif

#endif
