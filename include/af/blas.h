/********************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>

#ifdef __cplusplus
namespace af
{
    class array;
    /**
       C++ Interface to multiply two matrices.

       \copydetails blas_func_matmul

       `optLhs` and `optRhs` can only be one of \ref AF_MAT_NONE,
       \ref AF_MAT_TRANS, \ref AF_MAT_CTRANS.

       This function is not supported in GFOR.

       \note <b>The following applies for Sparse-Dense matrix multiplication.</b>
       \note This function can be used with one sparse input. The sparse input
             must always be the \p lhs and the dense matrix must be \p rhs.
       \note The sparse array can only be of \ref AF_STORAGE_CSR format.
       \note The returned array is always dense.
       \note \p optLhs an only be one of \ref AF_MAT_NONE, \ref AF_MAT_TRANS,
             \ref AF_MAT_CTRANS.
       \note \p optRhs can only be \ref AF_MAT_NONE.

       \param[in] lhs    input array on the left-hand side
       \param[in] rhs    input array on the right-hand side
       \param[in] optLhs transpose the left-hand side prior to multiplication
       \param[in] optRhs transpose the right-hand side prior to multiplication
       \return    `lhs` * `rhs`

       \ingroup blas_func_matmul
    */
    AFAPI array matmul(const array &lhs, const array &rhs,
                       const matProp optLhs = AF_MAT_NONE,
                       const matProp optRhs = AF_MAT_NONE);

    /**
       C++ Interface to multiply two matrices.
       The second matrix will be transposed.

       \copydetails blas_func_matmul

       This function is not supported in GFOR.

       \param[in] lhs input array on the left-hand side
       \param[in] rhs input array on the right-hand side
       \return    `lhs` * transpose(`rhs`)

       \ingroup blas_func_matmul
    */
    AFAPI array matmulNT(const array &lhs, const array &rhs);

    /**
       C++ Interface to multiply two matrices.
       The first matrix will be transposed.

       \copydetails blas_func_matmul

       This function is not supported in GFOR.

       \param[in] lhs input array on the left-hand side
       \param[in] rhs input array on the right-hand side
       \return    transpose(`lhs`) * `rhs`

       \ingroup blas_func_matmul
    */
    AFAPI array matmulTN(const array &lhs, const array &rhs);

    /**
       C++ Interface to multiply two matrices.
       Both matrices will be transposed.

       \copydetails blas_func_matmul

       This function is not supported in GFOR.

       \param[in] lhs input array on the left-hand side
       \param[in] rhs input array on the right-hand side
       \return    transpose(`lhs`) * transpose(`rhs`)

       \ingroup blas_func_matmul
    */
    AFAPI array matmulTT(const array &lhs, const array &rhs);

    /**
       C++ Interface to chain multiply three matrices.

       The matrix multiplications are done in a way to reduce temporary memory.

       This function is not supported in GFOR.

       \param[in] a The first array
       \param[in] b The second array
       \param[in] c The third array
       \return    a x b x c

       \ingroup blas_func_matmul
    */
    AFAPI array matmul(const array &a, const array &b, const array &c);


    /**
       C++ Interface to chain multiply three matrices.

       The matrix multiplications are done in a way to reduce temporary memory.

       This function is not supported in GFOR.

       \param[in] a The first array
       \param[in] b The second array
       \param[in] c The third array
       \param[in] d The fourth array
       \returns   a x b x c x d

       \ingroup blas_func_matmul
    */
    AFAPI array matmul(const array &a, const array &b, const array &c, const array &d);

#if AF_API_VERSION >= 35
    /**
        C++ Interface to compute the dot product.

        Scalar dot product between two vectors, also referred to as the inner
        product.

        \code
          // compute scalar dot product
          array x = randu(100), y = randu(100);

          af_print(dot(x, y));
          // OR
          printf("%f\n", dot<float>(x, y));
        \endcode

       Parameters `optLhs` and `optRhs` can only be one of \ref AF_MAT_NONE or
       \ref AF_MAT_CONJ. The conjugate dot product can be computed by setting
       `optLhs = AF_MAT_CONJ` and `optRhs = AF_MAT_NONE`.

       This function is not supported in GFOR.

        \tparam    T      type of the output
        \param[in] lhs    input array on the left-hand side
        \param[in] rhs    input array on the right-hand side
        \param[in] optLhs `lhs` options, only \ref AF_MAT_NONE and \ref
                          AF_MAT_CONJ are supported
        \param[in] optRhs `rhs` options, only \ref AF_MAT_NONE and \ref
                          AF_MAT_CONJ are supported
        \return    dot product of `lhs` and `rhs`

        \ingroup blas_func_dot
    */
    template <typename T>
    T dot(const array &lhs, const array &rhs,
          const matProp optLhs = AF_MAT_NONE,
          const matProp optRhs = AF_MAT_NONE);
#endif

    /// \ingroup blas_func_dot
    AFAPI array dot(const array &lhs, const array &rhs,
                    const matProp optLhs = AF_MAT_NONE,
                    const matProp optRhs = AF_MAT_NONE);

    /**
        C++ Interface to transpose a matrix.

        \param[in] in        input array
        \param[in] conjugate if true, conjugate transposition is performed
        \return    transpose

        \ingroup blas_func_transpose
    */
    AFAPI array transpose(const array &in, const bool conjugate = false);

    /**
        C++ Interface to transpose a matrix in-place.

        \param[in,out] in        input array to be transposed in-place
        \param[in]     conjugate if true, conjugate transposition is performed

        \ingroup blas_func_transpose
    */
    AFAPI void transposeInPlace(array &in, const bool conjugate = false);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if AF_API_VERSION >= 37
    /**
        C Interface to multiply two matrices.

        This provides an interface to the BLAS level 3 general matrix multiply
        (GEMM) of two \ref af_array objects, which is generally defined as:

        \f[
        C = \alpha * opA(A)opB(B) + \beta * C
        \f]

        where \f$\alpha\f$ (\p alpha) and \f$\beta\f$ (\p beta) are both scalars;
        \f$A\f$ and \f$B\f$ are the matrix multiply operands; and \f$opA\f$ and
        \f$opB\f$ are noop (if \p AF_MAT_NONE) or transpose (if \p AF_MAT_TRANS)
        operations on \f$A\f$ or \f$B\f$ before the actual GEMM operation. Batched
        GEMM is supported if at least either \f$A\f$ or \f$B\f$ have more than
        two dimensions (see \ref af::matmul for more details on broadcasting).
        However, only one \p alpha and one \p beta can be used for all of the
        batched matrix operands.

        The \ref af_array that \p out points to can be used both as an input and
        output. An allocation will be performed if you pass a null \ref af_array
        handle (i.e. `af_array c = 0;`). If a valid \ref af_array is passed as
        \f$C\f$, the operation will be performed on that \ref af_array itself. The C
        \ref af_array must be the correct type and shape; otherwise, an error will
        be thrown.

        \note Passing an af_array that has not been initialized to the C array
        is will cause undefined behavior.

        This example demonstrates the usage of the af_gemm function on two
        matrices. The \f$C\f$ \ref af_array handle is initialized to zero here,
        so \ref af_gemm will perform an allocation.

        \snippet test/blas.cpp ex_af_gemm_alloc

        The following example shows how you can write to a previously allocated \ref
        af_array using the \ref af_gemm call. Here we are going to use the \ref
        af_array s from the previous example and index into the first slice. Only
        the first slice of the original \f$C\f$ af_array will be modified by this
        operation.

        \snippet test/blas.cpp ex_af_gemm_overwrite

        \param[in,out] C     `A` * `B` = `C`
        \param[in]     opA   operation to perform on A before the multiplication
        \param[in]     opB   operation to perform on B before the multiplication
        \param[in]     alpha alpha value; must be the same type as `A` and `B`
        \param[in]     A     input array on the left-hand side
        \param[in]     B     input array on the right-hand side
        \param[in]     beta  beta value; must be the same type as `A` and `B`
        \return        \ref AF_SUCCESS, if function returns successfully, else
                       an \ref af_err code is given

        \ingroup blas_func_matmul
    */
    AFAPI af_err af_gemm(af_array *C, const af_mat_prop opA, const af_mat_prop opB,
                         const void *alpha, const af_array A, const af_array B,
                         const void *beta);
#endif

    /**
        C Interface to multiply two matrices.

        Performs matrix multiplication on two arrays.

        \note <b> The following applies for Sparse-Dense matrix multiplication.</b>
        \note This function can be used with one sparse input. The sparse input
              must always be the \p lhs and the dense matrix must be \p rhs.
        \note The sparse array can only be of \ref AF_STORAGE_CSR format.
        \note The returned array is always dense.
        \note \p optLhs an only be one of \ref AF_MAT_NONE, \ref AF_MAT_TRANS,
              \ref AF_MAT_CTRANS.
        \note \p optRhs can only be \ref AF_MAT_NONE.

        \param[out] out    `lhs` * `rhs` = `out`
        \param[in]  lhs    input array on the left-hand side
        \param[in]  rhs    input array on the right-hand side
        \param[in]  optLhs transpose `lhs` before the function is performed
        \param[in]  optRhs transpose `rhs` before the function is performed
        \return     \ref AF_SUCCESS, if function returns successfully, else
                    an \ref af_err code is given

        \ingroup blas_func_matmul
     */
    AFAPI af_err af_matmul( af_array *out ,
                            const af_array lhs, const af_array rhs,
                            const af_mat_prop optLhs, const af_mat_prop optRhs);

    /**
        C Interface to compute the dot product.

        Scalar dot product between two vectors, also referred to as the inner
        product.

        \code
          // compute scalar dot product
          array x = randu(100), y = randu(100);
          print(dot<float>(x,y));
        \endcode

        \param[out] out    dot product of `lhs` and `rhs`
        \param[in]  lhs    input array on the left-hand side
        \param[in]  rhs    input array on the right-hand side
        \param[in]  optLhs `lhs` options, only \ref AF_MAT_NONE and \ref
                           AF_MAT_CONJ are supported
        \param[in]  optRhs `rhs` options, only \ref AF_MAT_NONE and \ref
                           AF_MAT_CONJ are supported
        \return     \ref AF_SUCCESS, if function returns successfully, else
                    an \ref af_err code is given

        \ingroup blas_func_dot
    */
    AFAPI af_err af_dot(af_array *out,
                        const af_array lhs, const af_array rhs,
                        const af_mat_prop optLhs, const af_mat_prop optRhs);

#if AF_API_VERSION >= 35
    /**
        C Interface to compute the dot product, scalar result returned on host.

        Scalar dot product between two vectors. Also referred to as the inner
        product. Returns the result as a host scalar.

        \param[out] real   real component of the dot product
        \param[out] imag   imaginary component of the dot product
        \param[in]  lhs    input array on the left-hand side
        \param[in]  rhs    input array on the right-hand side
        \param[in]  optLhs `lhs` options, only \ref AF_MAT_NONE and \ref
                           AF_MAT_CONJ are supported
        \param[in]  optRhs `rhs` options, only \ref AF_MAT_NONE and \ref
                           AF_MAT_CONJ are supported
        \return     \ref AF_SUCCESS, if function returns successfully, else
                    an \ref af_err code is given

        \ingroup blas_func_dot
    */
    AFAPI af_err af_dot_all(double *real, double *imag,
                            const af_array lhs, const af_array rhs,
                            const af_mat_prop optLhs, const af_mat_prop optRhs);
#endif

    /**
        C Interface to transpose a matrix.

        \param[out] out       transpose
        \param[in]  in        input array
        \param[in]  conjugate if true, conjugate transposition is performed
        \return     \ref AF_SUCCESS, if function returns successfully, else
                    an \ref af_err code is given

        \ingroup blas_func_transpose
    */
    AFAPI af_err af_transpose(af_array *out, af_array in, const bool conjugate);

    /**
        C Interface to transpose a matrix in-place.

        \param[in,out] in        input array to be transposed in-place
        \param[in]     conjugate if true, conjugate transposition is performed
        \return        \ref AF_SUCCESS, if function returns successfully, else
                       an \ref af_err code is given

        \ingroup blas_func_transpose
    */
    AFAPI af_err af_transpose_inplace(af_array in, const bool conjugate);


#ifdef __cplusplus
}
#endif
