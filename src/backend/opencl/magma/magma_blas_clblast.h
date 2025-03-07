/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <complex>

#include <common/defines.hpp>
#include <common/half.hpp>

#include <clblast.h>
#include <err_clblast.hpp>

// Convert MAGMA constants to CLBlast constants
clblast::Layout clblast_order_const(magma_order_t order);
clblast::Transpose clblast_trans_const(magma_trans_t trans);
clblast::Triangle clblast_uplo_const(magma_uplo_t uplo);
clblast::Diagonal clblast_diag_const(magma_diag_t diag);
clblast::Side clblast_side_const(magma_side_t side);

// Error checking
#define OPENCL_BLAS_CHECK CLBLAST_CHECK

// Transposing
#define OPENCL_BLAS_TRANS_T clblast::Transpose  // the type
#define OPENCL_BLAS_NO_TRANS clblast::Transpose::kNo
#define OPENCL_BLAS_TRANS clblast::Transpose::kYes
#define OPENCL_BLAS_CONJ_TRANS clblast::Transpose::kConjugate

// Triangles
#define OPENCL_BLAS_TRIANGLE_T clblast::Triangle  // the type
#define OPENCL_BLAS_TRIANGLE_UPPER clblast::Triangle::kUpper
#define OPENCL_BLAS_TRIANGLE_LOWER clblast::Triangle::kLower

// Sides
#define OPENCL_BLAS_SIDE_RIGHT clblast::Side::kRight
#define OPENCL_BLAS_SIDE_LEFT clblast::Side::kLeft

// Unit or non-unit diagonal
#define OPENCL_BLAS_UNIT_DIAGONAL clblast::Diagonal::kUnit
#define OPENCL_BLAS_NON_UNIT_DIAGONAL clblast::Diagonal::kNonUnit

// Defines type conversions from ArrayFire (OpenCL) to CLBlast (C++ std)
template<typename T>
struct CLBlastType {
    using Type = T;
};
template<>
struct CLBlastType<cfloat> {
    using Type = std::complex<float>;
};
template<>
struct CLBlastType<cdouble> {
    using Type = std::complex<double>;
};
template<>
struct CLBlastType<arrayfire::common::half> {
    using Type = cl_half;
};

// Converts a constant from ArrayFire types (OpenCL) to CLBlast types (C++ std)
template<typename T>
typename CLBlastType<T>::Type inline toCLBlastConstant(const T val);

// Specializations of the above function
template<>
float inline toCLBlastConstant(const float val) {
    return val;
}
template<>
double inline toCLBlastConstant(const double val) {
    return val;
}
template<>
cl_half inline toCLBlastConstant(const arrayfire::common::half val) {
    cl_half out;
    memcpy(&out, &val, sizeof(cl_half));
    return out;
}
template<>
std::complex<float> inline toCLBlastConstant(cfloat val) {
    return {val.s[0], val.s[1]};
}
template<>
std::complex<double> inline toCLBlastConstant(cdouble val) {
    return {val.s[0], val.s[1]};
}

// Conversions to CLBlast basic types
template<typename T>
struct CLBlastBasicType {
    using Type = T;
};
template<>
struct CLBlastBasicType<arrayfire::common::half> {
    using Type = cl_half;
};
template<>
struct CLBlastBasicType<cfloat> {
    using Type = float;
};
template<>
struct CLBlastBasicType<cdouble> {
    using Type = double;
};

// Initialization of the OpenCL BLAS library
// Only meant to be once and from constructor
// of DeviceManager singleton
// DONT'T CALL FROM ANY OTHER LOCATION
inline void gpu_blas_init() {
    // Nothing to do here for CLBlast
}

// tear down of the OpenCL BLAS library
// Only meant to be called from destructor
// of DeviceManager singleton
// DONT'T CALL FROM ANY OTHER LOCATION
inline void gpu_blas_deinit() {
    // Nothing to do here for CLBlast
}

template<typename T>
struct gpu_blas_gemm_func {
    clblast::StatusCode operator()(
        const clblast::Transpose a_transpose,
        const clblast::Transpose b_transpose, const size_t m, const size_t n,
        const size_t k, const T alpha, const cl_mem a_buffer,
        const size_t a_offset, const size_t a_ld, const cl_mem b_buffer,
        const size_t b_offset, const size_t b_ld, const T beta, cl_mem c_buffer,
        const size_t c_offset, const size_t c_ld, cl_uint num_queues,
        cl_command_queue *queues, cl_uint num_wait_events,
        const cl_event wait_events, cl_event *events) {
        UNUSED(wait_events);
        assert(num_queues == 1);
        assert(num_wait_events == 0);
        const auto alpha_clblast = toCLBlastConstant(alpha);
        const auto beta_clblast  = toCLBlastConstant(beta);
        return clblast::Gemm(
            clblast::Layout::kColMajor, a_transpose, b_transpose, m, n, k,
            alpha_clblast, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld,
            beta_clblast, c_buffer, c_offset, c_ld, queues, events);
    }
};

template<typename T>
struct gpu_blas_gemv_func {
    clblast::StatusCode operator()(
        const clblast::Transpose a_transpose, const size_t m, const size_t n,
        const T alpha, const cl_mem a_buffer, const size_t a_offset,
        const size_t a_ld, const cl_mem x_buffer, const size_t x_offset,
        const size_t x_inc, const T beta, cl_mem y_buffer,
        const size_t y_offset, const size_t y_inc, cl_uint num_queues,
        cl_command_queue *queues, cl_uint num_wait_events,
        const cl_event *wait_events, cl_event *events) {
        UNUSED(wait_events);
        assert(num_queues == 1);
        assert(num_wait_events == 0);
        const auto alpha_clblast = toCLBlastConstant(alpha);
        const auto beta_clblast  = toCLBlastConstant(beta);
        return clblast::Gemv(clblast::Layout::kColMajor, a_transpose, m, n,
                             alpha_clblast, a_buffer, a_offset, a_ld, x_buffer,
                             x_offset, x_inc, beta_clblast, y_buffer, y_offset,
                             y_inc, queues, events);
    }
};

template<typename T>
struct gpu_blas_trmm_func {
    clblast::StatusCode operator()(
        const clblast::Side side, const clblast::Triangle triangle,
        const clblast::Transpose a_transpose, const clblast::Diagonal diagonal,
        const size_t m, const size_t n, const T alpha, const cl_mem a_buffer,
        const size_t a_offset, const size_t a_ld, cl_mem b_buffer,
        const size_t b_offset, const size_t b_ld, cl_uint num_queues,
        cl_command_queue *queues, cl_uint num_wait_events,
        const cl_event *wait_events, cl_event *events) {
        UNUSED(wait_events);
        assert(num_queues == 1);
        assert(num_wait_events == 0);
        const auto alpha_clblast = toCLBlastConstant(alpha);
        return clblast::Trmm(clblast::Layout::kColMajor, side, triangle,
                             a_transpose, diagonal, m, n, alpha_clblast,
                             a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld,
                             queues, events);
    }
};

template<typename T>
struct gpu_blas_trsm_func {
    clblast::StatusCode operator()(
        const clblast::Side side, const clblast::Triangle triangle,
        const clblast::Transpose a_transpose, const clblast::Diagonal diagonal,
        const size_t m, const size_t n, const T alpha, const cl_mem a_buffer,
        const size_t a_offset, const size_t a_ld, cl_mem b_buffer,
        const size_t b_offset, const size_t b_ld, cl_uint num_queues,
        cl_command_queue *queues, cl_uint num_wait_events,
        const cl_event *wait_events, cl_event *events) {
        UNUSED(wait_events);
        assert(num_queues == 1);
        assert(num_wait_events == 0);
        const auto alpha_clblast = toCLBlastConstant(alpha);
        return clblast::Trsm(clblast::Layout::kColMajor, side, triangle,
                             a_transpose, diagonal, m, n, alpha_clblast,
                             a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld,
                             queues, events);
    }
};

template<typename T>
struct gpu_blas_trsv_func {
    clblast::StatusCode operator()(
        const clblast::Triangle triangle, const clblast::Transpose a_transpose,
        const clblast::Diagonal diagonal, const size_t n, const cl_mem a_buffer,
        const size_t a_offset, const size_t a_ld, cl_mem x_buffer,
        const size_t x_offset, const size_t x_inc, cl_uint num_queues,
        cl_command_queue *queues, cl_uint num_wait_events,
        const cl_event *wait_events, cl_event *events) {
        UNUSED(wait_events);
        assert(num_queues == 1);
        assert(num_wait_events == 0);
        return clblast::Trsv<typename CLBlastType<T>::Type>(
            clblast::Layout::kColMajor, triangle, a_transpose, diagonal, n,
            a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, queues,
            events);
    }
};

template<typename T>
struct gpu_blas_herk_func {
    using BasicType = typename CLBlastBasicType<T>::Type;

    clblast::StatusCode operator()(
        const clblast::Triangle triangle, const clblast::Transpose a_transpose,
        const size_t n, const size_t k, const BasicType alpha,
        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
        const BasicType beta, cl_mem c_buffer, const size_t c_offset,
        const size_t c_ld, cl_uint num_queues, cl_command_queue *queues,
        cl_uint num_wait_events, const cl_event *wait_events,
        cl_event *events) {
        UNUSED(wait_events);
        assert(num_queues == 1);
        assert(num_wait_events == 0);
        const auto alpha_clblast = toCLBlastConstant(alpha);
        const auto beta_clblast  = toCLBlastConstant(beta);
        return clblast::Herk(clblast::Layout::kColMajor, triangle, a_transpose,
                             n, k, alpha_clblast, a_buffer, a_offset, a_ld,
                             beta_clblast, c_buffer, c_offset, c_ld, queues,
                             events);
    }
};

// Run syrk when calling non-complex herk function (specialisation of the above
// for 'float')
template<>
struct gpu_blas_herk_func<float> {
    clblast::StatusCode operator()(
        const clblast::Triangle triangle, const clblast::Transpose a_transpose,
        const size_t n, const size_t k, const float alpha,
        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
        const float beta, cl_mem c_buffer, const size_t c_offset,
        const size_t c_ld, cl_uint num_queues, cl_command_queue *queues,
        cl_uint num_wait_events, const cl_event *wait_events,
        cl_event *events) {
        UNUSED(wait_events);
        assert(num_queues == 1);
        assert(num_wait_events == 0);
        const auto alpha_clblast = toCLBlastConstant(alpha);
        const auto beta_clblast  = toCLBlastConstant(beta);
        return clblast::Syrk(clblast::Layout::kColMajor, triangle, a_transpose,
                             n, k, alpha_clblast, a_buffer, a_offset, a_ld,
                             beta_clblast, c_buffer, c_offset, c_ld, queues,
                             events);
    }
};

// Run syrk when calling non-complex herk function (specialisation of the above
// for 'double')
template<>
struct gpu_blas_herk_func<double> {
    clblast::StatusCode operator()(
        const clblast::Triangle triangle, const clblast::Transpose a_transpose,
        const size_t n, const size_t k, const double alpha,
        const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
        const double beta, cl_mem c_buffer, const size_t c_offset,
        const size_t c_ld, cl_uint num_queues, cl_command_queue *queues,
        cl_uint num_wait_events, const cl_event *wait_events,
        cl_event *events) {
        UNUSED(wait_events);
        assert(num_queues == 1);
        assert(num_wait_events == 0);
        const auto alpha_clblast = toCLBlastConstant(alpha);
        const auto beta_clblast  = toCLBlastConstant(beta);
        return clblast::Syrk(clblast::Layout::kColMajor, triangle, a_transpose,
                             n, k, alpha_clblast, a_buffer, a_offset, a_ld,
                             beta_clblast, c_buffer, c_offset, c_ld, queues,
                             events);
    }
};

template<typename T>
struct gpu_blas_syrk_func {
    clblast::StatusCode operator()(
        const clblast::Triangle triangle, const clblast::Transpose a_transpose,
        const size_t n, const size_t k, const T alpha, const cl_mem a_buffer,
        const size_t a_offset, const size_t a_ld, const T beta, cl_mem c_buffer,
        const size_t c_offset, const size_t c_ld, cl_uint num_queues,
        cl_command_queue *queues, cl_uint num_wait_events,
        const cl_event *wait_events, cl_event *events) {
        UNUSED(wait_events);
        assert(num_queues == 1);
        assert(num_wait_events == 0);
        const auto alpha_clblast = toCLBlastConstant(alpha);
        const auto beta_clblast  = toCLBlastConstant(beta);
        return clblast::Syrk(clblast::Layout::kColMajor, triangle, a_transpose,
                             n, k, alpha_clblast, a_buffer, a_offset, a_ld,
                             beta_clblast, c_buffer, c_offset, c_ld, queues,
                             events);
    }
};
