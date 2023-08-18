/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/array.h>
#include <af/defines.h>

#ifdef __cplusplus
namespace af
{
#if AF_API_VERSION >= 31
    /**
       C++ Interface to perform singular value decomposition.

       \param[out] u  U
       \param[out] s  diagonal values of sigma (singular values of the input 
                      matrix)
       \param[out] vt V^H
       \param[in]  in input array

       \ingroup lapack_factor_func_svd
    */
    AFAPI void svd(array &u, array &s, array &vt, const array &in);
#endif

#if AF_API_VERSION >= 31
    /**
       C++ Interface to perform in-place singular value decomposition.

       This function minimizes memory usage if `in` is dispensable. Input array
       `in` is limited to arrays where `dim0` \f$\geq\f$ `dim1`.

       \param[out]   u  U
       \param[out]   s  diagonal values of sigma (singular values of the input
                        matrix)
       \param[out]   vt V^H
       \param[inout] in input array; contains random data after the operation                       this operation

       \ingroup lapack_factor_func_svd
    */
    AFAPI void svdInPlace(array &u, array &s, array &vt, array &in);
#endif

    /**
       C++ Interface to perform LU decomposition in packed format.

       This function is not supported in GFOR.

       \param[out] out           packed LU decomposition
       \param[out] pivot         permutation indices mapping the input to the
                                 decomposition
       \param[in]  in            input array
       \param[in]  is_lapack_piv specifies if the pivot is returned in original
                                 LAPACK compliant format

       \ingroup lapack_factor_func_lu
    */
    AFAPI void lu(array &out, array &pivot, const array &in, const bool is_lapack_piv=true);

    /**
       C++ Interface to perform LU decomposition.

       This function is not supported in GFOR.

       \param[out] lower lower triangular matrix of the LU decomposition
       \param[out] upper upper triangular matrix of the LU decomposition
       \param[out] pivot permutation indices mapping the input to the
                         decomposition
       \param[in]  in    input array

       \ingroup lapack_factor_func_lu
    */
    AFAPI void lu(array &lower, array &upper, array &pivot, const array &in);

    /**
       C++ Interface to perform in-place LU decomposition.

       This function is not supported in GFOR.

       \param[out]   pivot         permutation indices mapping the input to the
                                   decomposition
       \param[inout] in            input array on entry; packed LU
                                   decomposition on exit
       \param[in]    is_lapack_piv specifies if the pivot is returned in
                                   original LAPACK-compliant format

       \ingroup lapack_factor_func_lu
    */
    AFAPI void luInPlace(array &pivot, array &in, const bool is_lapack_piv=true);

    /**
       C++ Interface to perform QR decomposition in packed format.

       This function is not supported in GFOR.

       \param[out] out packed QR decomposition
       \param[out] tau additional information needed for unpacking the data
       \param[in]  in  input array

       \ingroup lapack_factor_func_qr
    */
    AFAPI void qr(array &out, array &tau, const array &in);

    /**
       C++ Interface to perform QR decomposition.

       This function is not supported in GFOR.

       \param[out] q   orthogonal matrix from QR decomposition
       \param[out] r   upper triangular matrix from QR decomposition
       \param[out] tau additional information needed for solving a
                       least-squares problem using `q` and `r`
       \param[in]  in  input array

       \ingroup lapack_factor_func_qr
    */
    AFAPI void qr(array &q, array &r, array &tau, const array &in);

    /**
       C++ Interface to perform QR decomposition.

       This function is not supported in GFOR.

       \param[out]   tau additional information needed for unpacking the data
       \param[inout] in  input array on entry; packed QR decomposition on exit

       \ingroup lapack_factor_func_qr
    */
    AFAPI void qrInPlace(array &tau, array &in);

    /**
       C++ Interface to perform Cholesky decomposition.

       Multiplying `out` with its conjugate transpose reproduces the input
       `in`.
       
       The input must be positive definite.
       
       This function is not supported in GFOR.

       \param[out] out      triangular matrix; 
       \param[in]  in       input matrix
       \param[in]  is_upper boolean determining if `out` is upper or lower
                            triangular
       \returns    `0` if cholesky decomposition passes; if not, it returns the
                   rank at which the decomposition fails

       \ingroup lapack_factor_func_cholesky
    */
    AFAPI int cholesky(array &out, const array &in, const bool is_upper = true);

    /**
       C++ Interface to perform in-place Cholesky decomposition.

       The input must be positive definite.

       This function is not supported in GFOR.

       \param[inout] in       input matrix on entry; triangular matrix on exit
       \param[in]    is_upper boolean determining if `in` is upper or lower
                              triangular
       \returns      `0` if cholesky decomposition passes; if not, it returns
                     the rank at which the decomposition fails

       \ingroup lapack_factor_func_cholesky
    */
    AFAPI int choleskyInPlace(array &in, const bool is_upper = true);

    /**
       C++ Interface to solve a system of equations.

       The `options` parameter must be one of \ref AF_MAT_NONE,
       \ref AF_MAT_LOWER or \ref AF_MAT_UPPER.

       This function is not supported in GFOR.

       \param[in] a       coefficient matrix
       \param[in] b       measured values
       \param[in] options determines various properties of matrix `a`
       \returns   `x`, the matrix of unknown variables

       \ingroup lapack_solve_func_gen
    */
    AFAPI array solve(const array &a, const array &b, const matProp options = AF_MAT_NONE);

    /**
       C++ Interface to solve a system of equations.

       The `options` parameter currently must be \ref AF_MAT_NONE.

       This function is not supported in GFOR.

       \param[in] a       packed LU decomposition of the coefficient matrix
       \param[in] piv     pivot array from the packed LU decomposition of the
                          coefficient matrix
       \param[in] b       measured values
       \param[in] options determines various properties of matrix `a`
       \returns   `x`, the matrix of unknown variables

       \ingroup lapack_solve_lu_func_gen
    */
    AFAPI array solveLU(const array &a, const array &piv,
                        const array &b, const matProp options = AF_MAT_NONE);

    /**
       C++ Interface to invert a matrix.

       The `options` parameter currently must be \ref AF_MAT_NONE.

       This function is not supported in GFOR.

       \param[in] in      input matrix
       \param[in] options determines various properties of matrix `in`
       \returns   inverse matrix

       \ingroup lapack_ops_func_inv
    */
    AFAPI array inverse(const array &in, const matProp options = AF_MAT_NONE);

#if AF_API_VERSION >= 37
    /**
       C++ Interface to pseudo-invert (Moore-Penrose) a matrix.

       Currently uses the SVD-based approach.

       Parameter `tol` is not the actual lower threshold, but it is passed in
       as a parameter to the calculation of the actual threshold relative to
       the shape and contents of `in`.
       
       This function is not supported in GFOR.

       \param[in] in      input matrix
       \param[in] tol     defines the lower threshold for singular values from
                          SVD
       \param[in] options must be AF_MAT_NONE (more options might be supported
                          in the future)
       \returns   pseudo-inverse matrix

       \ingroup lapack_ops_func_pinv
    */
    AFAPI array pinverse(const array &in, const double tol=1E-6,
                         const matProp options = AF_MAT_NONE);
#endif

    /**
       C++ Interface to find the rank of a matrix.

       \param[in] in  input matrix
       \param[in] tol tolerance value
       \returns   rank

       \ingroup lapack_ops_func_rank
    */
    AFAPI unsigned rank(const array &in, const double tol=1E-5);

    /**
       C++ Interface to find the determinant of a matrix.

       \param[in] in input matrix
       \returns   determinant

       \ingroup lapack_ops_func_det
    */
    template<typename T> T det(const array &in);

    /**
       C++ Interface to find the norm of a matrix.

       \param[in] in   input matrix
       \param[in] type \ref af::normType. Default: \ref AF_NORM_VECTOR_1
       \param[in] p    value of P when `type` is \ref AF_NORM_VECTOR_P or
                       \ref AF_NORM_MATRIX_L_PQ, else ignored
       \param[in] q    value of Q when `type` is \ref AF_NORM_MATRIX_L_PQ, else
                       ignored
       \returns   norm

       \ingroup lapack_ops_func_norm
    */
    AFAPI double norm(const array &in, const normType type=AF_NORM_EUCLID,
                      const double p=1, const double q=1);

#if AF_API_VERSION >= 33
    /**
       Returns true if ArrayFire is compiled with LAPACK support.

       \returns true if LAPACK support is available; false otherwise

       \ingroup lapack_helper_func_available
    */
    AFAPI bool isLAPACKAvailable();
#endif

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if AF_API_VERSION >= 31
    /**
       C Interface to perform singular value decomposition.

       \param[out] u  U
       \param[out] s  diagonal values of sigma (singular values of the input
                      matrix)
       \param[out] vt V^H
       \param[in]  in input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup lapack_factor_func_svd
    */
    AFAPI af_err af_svd(af_array *u, af_array *s, af_array *vt, const af_array in);
#endif

#if AF_API_VERSION >= 31
    /**
       C Interface to perform in-place singular value decomposition.

       This function minimizes memory usage if `in` is dispensable. Input array
       `in` is limited to arrays where `dim0` \f$\geq\f$ `dim1`.

       \param[out]   u  U
       \param[out]   s  diagonal values of sigma (singular values of the input
                        matrix)
       \param[out]   vt V^H
       \param[inout] in input array; contains random data after the operation                       this operation
       \return       \ref AF_SUCCESS, if function returns successfully, else
                     an \ref af_err code is given

       \ingroup lapack_factor_func_svd
    */
    AFAPI af_err af_svd_inplace(af_array *u, af_array *s, af_array *vt, af_array in);
#endif

    /**
       C Interface to perform LU decomposition.

       \param[out] lower lower triangular matrix of the LU decomposition
       \param[out] upper upper triangular matrix of the LU decomposition
       \param[out] pivot permutation indices mapping the input to the
                         decomposition
       \param[in]  in    input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup lapack_factor_func_lu
    */
    AFAPI af_err af_lu(af_array *lower, af_array *upper, af_array *pivot, const af_array in);

    /**
       C Interface to perform in-place LU decomposition.

       This function is not supported in GFOR.

       \param[out]   pivot         permutation indices mapping the input to the
                                   decomposition
       \param[inout] in            input array on entry; packed LU
                                   decomposition on exit
       \param[in]    is_lapack_piv specifies if the pivot is returned in
                                   original LAPACK-compliant format
       \return       \ref AF_SUCCESS, if function returns successfully, else
                     an \ref af_err code is given

       \ingroup lapack_factor_func_lu
    */
    AFAPI af_err af_lu_inplace(af_array *pivot, af_array in, const bool is_lapack_piv);

    /**
       C Interface to perform QR decomposition.

       This function is not supported in GFOR.

       \param[out] q   orthogonal matrix from QR decomposition
       \param[out] r   upper triangular matrix from QR decomposition
       \param[out] tau additional information needed for solving a
                       least-squares problem using `q` and `r`
       \param[in]  in  input array
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup lapack_factor_func_qr
    */
    AFAPI af_err af_qr(af_array *q, af_array *r, af_array *tau, const af_array in);

    /**
       C Interface to perform QR decomposition.

       This function is not supported in GFOR.

       \param[out]   tau additional information needed for unpacking the data
       \param[inout] in  input array on entry; packed QR decomposition on exit
       \return       \ref AF_SUCCESS, if function returns successfully, else
                     an \ref af_err code is given

       \ingroup lapack_factor_func_qr
    */
    AFAPI af_err af_qr_inplace(af_array *tau, af_array in);

    /**
       C Interface to perform Cholesky decomposition.

       Multiplying `out` with its conjugate transpose reproduces the input
       `in`.

       The input must be positive definite.

       \param[out] out      triangular matrix;
       \param[out] info     `0` if cholesky decomposition passes; if not, it
                            returns the rank at which the decomposition fails
       \param[in]  in       input matrix
       \param[in]  is_upper boolean determining if `out` is upper or lower
                            triangular
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup lapack_factor_func_cholesky
    */
    AFAPI af_err af_cholesky(af_array *out, int *info, const af_array in, const bool is_upper);

    /**
       C Interface to perform in-place Cholesky decomposition.

       The input must be positive definite.

       \param[out]   info     `0` if cholesky decomposition passes; if not, it
                              returns the rank at which the decomposition fails
       \param[inout] in       input matrix on entry; triangular matrix on exit
       \param[in]    is_upper boolean determining if `in` is upper or lower
                              triangular
       \return       \ref AF_SUCCESS, if function returns successfully, else
                     an \ref af_err code is given

       \ingroup lapack_factor_func_cholesky
    */
    AFAPI af_err af_cholesky_inplace(int *info, af_array in, const bool is_upper);

    /**
       C Interface to solve a system of equations.

       The `options` parameter must be one of \ref AF_MAT_NONE,
       \ref AF_MAT_LOWER or \ref AF_MAT_UPPER.

       This function is not supported in GFOR.

       \param[out] x       matrix of unknown variables
       \param[in]  a       coefficient matrix
       \param[in]  b       measured values
       \param[in]  options determines various properties of matrix `a`
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup lapack_solve_func_gen
    */
    AFAPI af_err af_solve(af_array *x, const af_array a, const af_array b,
                          const af_mat_prop options);

    /**
       C Interface to solve a system of equations.

       The `options` parameter currently must be \ref AF_MAT_NONE.

       \param[out] x       matrix of unknown variables
       \param[in]  a       packed LU decomposition of the coefficient matrix
       \param[in]  piv     pivot array from the packed LU decomposition of the
                           coefficient matrix
       \param[in]  b       measured values
       \param[in]  options determines various properties of matrix `a`
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup lapack_solve_lu_func_gen
    */
    AFAPI af_err af_solve_lu(af_array *x, const af_array a, const af_array piv,
                             const af_array b, const af_mat_prop options);

    /**
       C Interface to invert a matrix.

       The `options` parameter currently must be \ref AF_MAT_NONE.

       \param[out] out     inverse matrix
       \param[in]  in      input matrix
       \param[in]  options determines various properties of matrix `in`
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup lapack_ops_func_inv
    */
    AFAPI af_err af_inverse(af_array *out, const af_array in, const af_mat_prop options);

#if AF_API_VERSION >= 37
    /**
       C Interface to pseudo-invert (Moore-Penrose) a matrix.

       Currently uses the SVD-based approach.

       Parameter `tol` is not the actual lower threshold, but it is passed in
       as a parameter to the calculation of the actual threshold relative to
       the shape and contents of `in`.

       Suggested parameters for `tol`:  1e-6 for single precision and 1e-12 for
       double precision.

       \param[out] out     pseudo-inverse matrix
       \param[in]  in      input matrix
       \param[in]  tol     defines the lower threshold for singular values from
                           SVD
       \param[in]  options must be AF_MAT_NONE (more options might be supported
                           in the future)
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup lapack_ops_func_pinv
    */
    AFAPI af_err af_pinverse(af_array *out, const af_array in, const double tol,
                             const af_mat_prop options);
#endif

    /**
       C Interface to find the rank of a matrix.

       \param[out] rank rank
       \param[in]  in   input matrix
       \param[in]  tol  tolerance value
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup lapack_ops_func_rank
    */
    AFAPI af_err af_rank(unsigned *rank, const af_array in, const double tol);

    /**
       C Interface to find the determinant of a matrix.

       \param[out] det_real real part of the determinant
       \param[out] det_imag imaginary part of the determinant
       \param[in]  in       input matrix
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup lapack_ops_func_det
    */
    AFAPI af_err af_det(double *det_real, double *det_imag, const af_array in);

    /**
       C Interface to find the norm of a matrix.

       \param[out] out  norm
       \param[in]  in   input matrix
       \param[in]  type \ref af::normType. Default: \ref AF_NORM_VECTOR_1
       \param[in]  p    value of P when `type` is \ref AF_NORM_VECTOR_P or
                        \ref AF_NORM_MATRIX_L_PQ, else ignored
       \param[in]  q    value of Q when `type` is \ref AF_NORM_MATRIX_L_PQ, else
                        ignored
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup lapack_ops_func_norm
    */
    AFAPI af_err af_norm(double *out, const af_array in, const af_norm_type type, const double p, const double q);

#if AF_API_VERSION >= 33
    /**
       Returns true if ArrayFire is compiled with LAPACK support.

       \param[out] out true if LAPACK support is available; false otherwise
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given; does not depend on the value
                   of `out`

       \ingroup lapack_helper_func_available
    */
    AFAPI af_err af_is_lapack_available(bool *out);
#endif


#ifdef __cplusplus
}
#endif
