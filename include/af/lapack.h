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

    /**
       C++ Interface for LU decomposition in packed format

       \param[out] out is the output array containing the packed LU decomposition
       \param[out] pivot will contain the permutation indices to map the input to the decomposition
       \param[in] in is the input matrix
       \param[in] is_lapack_piv specifies if the pivot is returned in original LAPACK compliant format

       \note This function is not supported in GFOR

       \ingroup lapack_factor_func_lu
    */
    AFAPI void lu(array &out, array &pivot, const array &in, const bool is_lapack_piv=true);

    /**
       C++ Interface for LU decomposition

       \param[out] lower will contain the lower triangular matrix of the LU decomposition
       \param[out] upper will contain the upper triangular matrix of the LU decomposition
       \param[out] pivot will contain the permutation indices to map the input to the decomposition
       \param[in] in is the input matrix

       \note This function is not supported in GFOR

       \ingroup lapack_factor_func_lu
    */
    AFAPI void lu(array &lower, array &upper, array &pivot, const array &in);

    /**
      C++ Interface for in place LU decomposition

      \param[out] pivot will contain the permutation indices to map the input to the decomposition
      \param[inout] in contains the input on entry, the packed LU decomposition on exit
      \param[in] is_lapack_piv specifies if the pivot is returned in original LAPACK compliant format

      \note This function is not supported in GFOR

      \ingroup lapack_factor_func_lu
    */
    AFAPI void luInPlace(array &pivot, array &in, const bool is_lapack_piv=true);

    /**
       C++ Interface for QR decomposition in packed format

       \param[out] out is the output array containing the packed QR decomposition
       \param[out] tau will contain additional information needed for unpacking the data
       \param[in] in is the input matrix

       \note This function is not supported in GFOR

       \ingroup lapack_factor_func_qr
    */
    AFAPI void qr(array &out, array &tau, const array &in);

    /**
       C++ Interface for QR decomposition

       \param[out] q is the orthogonal matrix from QR decomposition
       \param[out] r is the upper triangular matrix from QR decomposition
       \param[out] tau will contain additional information needed for solving a least squares problem using \p q and \p r
       \param[in] in is the input matrix

       \note This function is not supported in GFOR

       \ingroup lapack_factor_func_qr
    */
    AFAPI void qr(array &q, array &r, array &tau, const array &in);

    /**
       C++ Interface for QR decomposition

       \param[out] tau will contain additional information needed for unpacking the data
       \param[inout] in is the input matrix on entry. It contains packed QR decomposition on exit

       \note This function is not supported in GFOR

       \ingroup lapack_factor_func_qr
    */
    AFAPI void qrInPlace(array &tau, array &in);

    /**
       C++ Interface for cholesky decomposition

       \param[out] out contains the triangular matrix. Multiply \p out with its conjugate transpose reproduces the input \p in.
       \param[in] in is the input matrix
       \param[in] is_upper a boolean determining if \p out is upper or lower triangular

       \returns \p 0 if cholesky decomposition passes, if not it returns the rank at which the decomposition failed.

       \note The input matrix \b has to be a positive definite matrix, if it is not zero, the cholesky decomposition functions return a non-zero output.
       \note This function is not supported in GFOR

       \ingroup lapack_factor_func_cholesky
    */
    AFAPI int cholesky(array &out, const array &in, const bool is_upper = true);

    /**
       C++ Interface for in place cholesky decomposition

       \param[inout] in is the input matrix on entry. It contains the triangular matrix on exit.
       \param[in] is_upper a boolean determining if \p in is upper or lower triangular

       \returns \p 0 if cholesky decomposition passes, if not it returns the rank at which the decomposition failed.

       \note The input matrix \b has to be a positive definite matrix, if it is not zero, the cholesky decomposition functions return a non-zero output.
       \note This function is not supported in GFOR

       \ingroup lapack_factor_func_cholesky
    */
    AFAPI int choleskyInPlace(array &in, const bool is_upper = true);

    /**
       C++ Interface for solving a system of equations

       \param[in] a is the coefficient matrix
       \param[in] b is the measured values
       \param[in] options determining various properties of matrix \p a
       \returns \p x, the matrix of unknown variables

       \note \p options needs to be one of \ref AF_MAT_NONE, \ref AF_MAT_LOWER or \ref AF_MAT_UPPER
       \note This function is not supported in GFOR

       \ingroup lapack_solve_func_gen
    */
    AFAPI array solve(const array &a, const array &b, const matProp options = AF_MAT_NONE);


    /**
       C++ Interface for solving a system of equations

       \param[in] a is the output matrix from packed LU decomposition of the coefficient matrix
       \param[in] piv is the pivot array from packed LU decomposition of the coefficient matrix
       \param[in] b is the matrix of measured values
       \param[in] options determining various properties of matrix \p a
       \returns \p x, the matrix of unknown variables

       \ingroup lapack_solve_lu_func_gen

       \note \p options currently needs to be \ref AF_MAT_NONE
       \note This function is not supported in GFOR
    */
    AFAPI array solveLU(const array &a, const array &piv,
                        const array &b, const matProp options = AF_MAT_NONE);

    /**
       C++ Interface for inverting a matrix

       \param[in] in is input matrix
       \param[in] options determining various properties of matrix \p in
       \returns \p x, the inverse of the input matrix

       \note \p options currently needs to be \ref AF_MAT_NONE
       \note This function is not supported in GFOR

       \ingroup lapack_ops_func_inv
    */
    AFAPI array inverse(const array &in, const matProp options = AF_MAT_NONE);

    /**
       C++ Interface for finding the rank of a matrix

       \param[in] in is input matrix
       \param[in] tol is the tolerance value

       \returns the rank of the matrix

       \ingroup lapack_ops_func_rank
    */
    AFAPI unsigned rank(const array &in, const double tol=1E-5);

    /**
       C++ Interface for finding the determinant of a matrix

       \param[in] in is input matrix

       \returns the determinant of the matrix

       \ingroup lapack_ops_func_det
    */
    template<typename T> T det(const array &in);

    /**
       C++ Interface for norm of a matrix

       \param[in] in is the input matrix
       \param[in] type specifies the \ref af::normType. Default: \ref AF_NORM_VECTOR_1
       \param[in] p specifies the value of P when \p type is one of \ref AF_NORM_VECTOR_P, AF_NORM_MATRIX_L_PQ is used. It is ignored for other values of \p type
       \param[in] q specifies the value of Q when \p type is AF_NORM_MATRIX_L_PQ. This parameter is ignored if \p type is anything else

       \returns the norm of \p inbased on \p type

       \ingroup lapack_ops_func_norm
    */
    AFAPI double norm(const array &in, const normType type=AF_NORM_EUCLID,
                      const double p=1, const double q=1);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       C Interface for LU decomposition

       \param[out] lower will contain the lower triangular matrix of the LU decomposition
       \param[out] upper will contain the upper triangular matrix of the LU decomposition
       \param[out] pivot will contain the permutation indices to map the input to the decomposition
       \param[in] in is the input matrix

       \ingroup lapack_factor_func_lu
    */
    AFAPI af_err af_lu(af_array *lower, af_array *upper, af_array *pivot, const af_array in);

    /**
       C Interface for in place LU decomposition

       \param[out] pivot will contain the permutation indices to map the input to the decomposition
       \param[inout] in contains the input on entry, the packed LU decomposition on exit
       \param[in] is_lapack_piv specifies if the pivot is returned in original LAPACK compliant format

       \ingroup lapack_factor_func_lu
    */
    AFAPI af_err af_lu_inplace(af_array *pivot, af_array in, const bool is_lapack_piv);

    /**
       C Interface for QR decomposition

       \param[out] q is the orthogonal matrix from QR decomposition
       \param[out] r is the upper triangular matrix from QR decomposition
       \param[out] tau will contain additional information needed for solving a least squares problem using \p q and \p r
       \param[in] in is the input matrix

       \ingroup lapack_factor_func_qr
    */
    AFAPI af_err af_qr(af_array *q, af_array *r, af_array *tau, const af_array in);

    /**
       C Interface for QR decomposition

       \param[out] tau will contain additional information needed for unpacking the data
       \param[inout] in is the input matrix on entry. It contains packed QR decomposition on exit

       \ingroup lapack_factor_func_qr
    */
    AFAPI af_err af_qr_inplace(af_array *tau, af_array in);

    /**
       C++ Interface for cholesky decomposition

       \param[out] out contains the triangular matrix. Multiply \p out with it conjugate transpose reproduces the input \p in.
       \param[out] info is \p 0 if cholesky decomposition passes, if not it returns the rank at which the decomposition failed.
       \param[in] in is the input matrix
       \param[in] is_upper a boolean determining if \p out is upper or lower triangular

       \note The input matrix \b has to be a positive definite matrix, if it is not zero, the cholesky decomposition functions return a non zero output.

       \ingroup lapack_factor_func_cholesky
    */
    AFAPI af_err af_cholesky(af_array *out, int *info, const af_array in, const bool is_upper);

    /**
       C Interface for in place cholesky decomposition

       \param[out] info is \p 0 if cholesky decomposition passes, if not it returns the rank at which the decomposition failed.
       \param[inout] in is the input matrix on entry. It contains the triangular matrix on exit.
       \param[in] is_upper a boolean determining if \p in is upper or lower triangular

       \note The input matrix \b has to be a positive definite matrix, if it is not zero, the cholesky decomposition functions return a non zero output.

       \ingroup lapack_factor_func_cholesky
    */
    AFAPI af_err af_cholesky_inplace(int *info, af_array in, const bool is_upper);

    /**
       C Interface for solving a system of equations

       \param[out] x is the matrix of unknown variables
       \param[in] a is the coefficient matrix
       \param[in] b is the measured values
       \param[in] options determining various properties of matrix \p a

       \ingroup lapack_solve_func_gen

       \note \p options needs to be one of \ref AF_MAT_NONE, \ref AF_MAT_LOWER or \ref AF_MAT_UPPER
    */
    AFAPI af_err af_solve(af_array *x, const af_array a, const af_array b,
                          const af_mat_prop options);

    /**
       C Interface for solving a system of equations

       \param[out] x will contain the matrix of unknown variables
       \param[in] a is the output matrix from packed LU decomposition of the coefficient matrix
       \param[in] piv is the pivot array from packed LU decomposition of the coefficient matrix
       \param[in] b is the matrix of measured values
       \param[in] options determining various properties of matrix \p a

       \ingroup lapack_solve_lu_func_gen

       \note \p options currently needs to be \ref AF_MAT_NONE
       \note This function is not supported in GFOR
    */
    AFAPI af_err af_solve_lu(af_array *x, const af_array a, const af_array piv,
                             const af_array b, const af_mat_prop options);

    /**
       C Interface for inverting a matrix

       \param[out] out will contain the inverse of matrix \p in
       \param[in] in is input matrix
       \param[in] options determining various properties of matrix \p in

       \ingroup lapack_ops_func_inv

       \note currently options needs to be \ref AF_MAT_NONE
    */
    AFAPI af_err af_inverse(af_array *out, const af_array in, const af_mat_prop options);

    /**
       C Interface for finding the rank of a matrix

       \param[out] rank will contain the rank of \p in
       \param[in] in is input matrix
       \param[in] tol is the tolerance value

       \ingroup lapack_ops_func_rank
    */
    AFAPI af_err af_rank(unsigned *rank, const af_array in, const double tol);

    /**
       C Interface for finding the determinant of a matrix

       \param[out] det_real will contain the real part of the determinant of \p in
       \param[out] det_imag will contain the imaginary part of the determinant of \p in
       \param[in] in is input matrix

       \ingroup lapack_ops_func_det
    */
    AFAPI af_err af_det(double *det_real, double *det_imag, const af_array in);

    /**
       C Interface for norm of a matrix

       \param[out] out will contain the norm of \p in
       \param[in] in is the input matrix
       \param[in] type specifies the \ref af::normType. Default: \ref AF_NORM_VECTOR_1
       \param[in] p specifies the value of P when \p type is one of \ref AF_NORM_VECTOR_P,  AF_NORM_MATRIX_L_PQ is used. It is ignored for other values of \p type
       \param[in] q specifies the value of Q when \p type is AF_NORM_MATRIX_L_PQ. This parameter is ignored if \p type is anything else


       \ingroup lapack_ops_func_norm
    */
    AFAPI af_err af_norm(double *out, const af_array in, const af_norm_type type, const double p, const double q);


#ifdef __cplusplus
}
#endif
