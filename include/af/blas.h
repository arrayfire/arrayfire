/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/** \file blas.h
 *
 * Contains BLAS related functions
 *
 * Contains functions for basic BLAS functionallity
 */

#pragma once

#include <af/defines.h>

#ifdef __cplusplus
namespace af
{
    class array;
    /**
        \brief Matrix multiply of two arrays

        \copydetails blas_func_matmul

        \param[in] lhs The array object on the left hand side
        \param[in] rhs The array object on the right hand side
        \param[in] optLhs Transpose left hand side before the function is performed
        \param[in] optRhs Transpose right hand side before the function is performed
        \return The result of the matrix multiplication of lhs, rhs

        \note optLhs and optRhs can only be one of \ref AF_MAT_NONE, \ref
              AF_MAT_TRANS, \ref AF_MAT_CTRANS \note This function is not supported
              in GFOR

        \note <b> The following applies for Sparse-Dense matrix multiplication.</b>
        \note This function can be used with one sparse input. The sparse input
              must always be the \p lhs and the dense matrix must be \p rhs.
        \note The sparse array can only be of \ref AF_STORAGE_CSR format.
        \note The returned array is always dense.
        \note \p optLhs an only be one of \ref AF_MAT_NONE, \ref AF_MAT_TRANS,
              \ref AF_MAT_CTRANS.
        \note \p optRhs can only be \ref AF_MAT_NONE.

        \ingroup blas_func_matmul

     */
    AFAPI array matmul(const array &lhs, const array &rhs,
                       const matProp optLhs = AF_MAT_NONE,
                       const matProp optRhs = AF_MAT_NONE);

    /**
       \brief Matrix multiply of two arrays

       \copydetails blas_func_matmul

       \param[in] lhs The array object on the left hand side
       \param[in] rhs The array object on the right hand side
       \return The result of the matrix multiplication of \p lhs, transpose(\p rhs)

       \note This function is not supported in GFOR

       \ingroup blas_func_matmul
    */
    AFAPI array matmulNT(const array &lhs, const array &rhs);

    /**
       \brief Matrix multiply of two arrays

       \copydetails blas_func_matmul

       \param[in] lhs The array object on the left hand side
       \param[in] rhs The array object on the right hand side
       \return The result of the matrix multiplication of transpose(\p lhs), \p rhs

       \note This function is not supported in GFOR

       \ingroup blas_func_matmul
    */
    AFAPI array matmulTN(const array &lhs, const array &rhs);

    /**
       \brief Matrix multiply of two arrays

       \copydetails blas_func_matmul

       \param[in] lhs The array object on the left hand side
       \param[in] rhs The array object on the right hand side
       \return The result of the matrix multiplication of transpose(\p lhs), transpose(\p rhs)

       \note This function is not supported in GFOR

       \ingroup blas_func_matmul
    */
    AFAPI array matmulTT(const array &lhs, const array &rhs);

    /**
       \brief Chain 2 matrix multiplications

       The matrix multiplications are done in a way to reduce temporary memory

       \param[in] a The first array
       \param[in] b The second array
       \param[in] c The third array

       \returns out = a x b x c

       \note This function is not supported in GFOR

       \ingroup blas_func_matmul
    */
    AFAPI array matmul(const array &a, const array &b, const array &c);


    /**
       \brief Chain 3 matrix multiplications

       The matrix multiplications are done in a way to reduce temporary memory

       \param[in] a The first array
       \param[in] b The second array
       \param[in] c The third array
       \param[in] d The fourth array

       \returns out = a x b x c x d

       \note This function is not supported in GFOR

       \ingroup blas_func_matmul
    */
    AFAPI array matmul(const array &a, const array &b, const array &c, const array &d);

#if AF_API_VERSION >= 35
    /**
        \brief Dot Product

        Scalar dot product between two vectors. Also referred to as the inner
        product.

        \code
          // compute scalar dot product
          array x = randu(100),
          y = randu(100);

          af_print(dot(x, y));
          // OR
          printf("%f\n", dot<float>(x, y));

        \endcode

        \tparam T The type of the output
        \param[in] lhs The array object on the left hand side
        \param[in] rhs The array object on the right hand side
        \param[in] optLhs Options for lhs. Currently only \ref AF_MAT_NONE and
                  AF_MAT_CONJ are supported.
        \param[in] optRhs Options for rhs. Currently only \ref AF_MAT_NONE and
        AF_MAT_CONJ are supported \return The result of the dot product of lhs,
        rhs

        \note optLhs and optRhs can only be one of \ref AF_MAT_NONE or \ref
              AF_MAT_CONJ
        \note optLhs = AF_MAT_CONJ and optRhs = AF_MAT_NONE will run
              conjugate dot operation.
        \note This function is not supported in GFOR

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
        \brief C++ Interface for transposing a matrix

        \param[in] in an input matrix
        \param[in] conjugate if true, a conjugate transposition is performed
        \return the transposed matrix
        \ingroup blas_func_transpose
    */
    AFAPI array transpose(const array &in, const bool conjugate = false);

    /**
        \brief C++ Interface for transposing a matrix in-place

        \param[in,out] in the matrix to be transposed in-place
        \param[in] conjugate if true, a conjugate transposition is performed

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
        \brief BLAS general matrix multiply (GEMM) of two \ref af_array objects

        \details
        This provides a general interface to the BLAS level 3 general matrix
        multiply (GEMM), which is generally defined as:

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

        \param[in,out] C     Pointer to the output \ref af_array

        \param[in]     opA   Operation to perform on A before the multiplication

        \param[in]     opB   Operation to perform on B before the multiplication

        \param[in]     alpha The alpha value; must be the same type as \p lhs
                            and \p rhs

        \param[in]     A     Left-hand side operand

        \param[in]     B     Right-hand side operand

        \param[in]     beta  The beta value; must be the same type as \p lhs
                            and \p rhs

        \return AF_SUCCESS if the operation is successful.

        \ingroup blas_func_matmul
    */
    AFAPI af_err af_gemm(af_array *C, const af_mat_prop opA, const af_mat_prop opB,
                         const void *alpha, const af_array A, const af_array B,
                         const void *beta);
#endif

    /**
        \brief Matrix multiply of two \ref af_array

        \details Performs a matrix multiplication on two arrays (lhs, rhs).

        \param[out] out Pointer to the output \ref af_array
        \param[in] lhs A 2D matrix \ref af_array object
        \param[in] rhs A 2D matrix \ref af_array object
        \param[in] optLhs Transpose left hand side before the function is performed
        \param[in] optRhs Transpose right hand side before the function is performed

        \return AF_SUCCESS if the process is successful.

        \note <b> The following applies for Sparse-Dense matrix multiplication.</b>
        \note This function can be used with one sparse input. The sparse input
              must always be the \p lhs and the dense matrix must be \p rhs.
        \note The sparse array can only be of \ref AF_STORAGE_CSR format.
        \note The returned array is always dense.
        \note \p optLhs an only be one of \ref AF_MAT_NONE, \ref AF_MAT_TRANS,
              \ref AF_MAT_CTRANS.
        \note \p optRhs can only be \ref AF_MAT_NONE.

        \ingroup blas_func_matmul
     */
    AFAPI af_err af_matmul( af_array *out ,
                            const af_array lhs, const af_array rhs,
                            const af_mat_prop optLhs, const af_mat_prop optRhs);


    /**
        Scalar dot product between two vectors.  Also referred to as the inner
        product.

        \code
        // compute scalar dot product
        array x = randu(100), y = randu(100);
        print(dot<float>(x,y));
        \endcode

        \param[out] out The array object with the result of the dot operation
        \param[in] lhs The array object on the left hand side
        \param[in] rhs The array object on the right hand side
        \param[in] optLhs Options for lhs. Currently only \ref AF_MAT_NONE and
                   AF_MAT_CONJ are supported.
        \param[in] optRhs Options for rhs. Currently only \ref AF_MAT_NONE and AF_MAT_CONJ are supported
        \return AF_SUCCESS if the process is successful.

        \ingroup blas_func_dot
    */
    AFAPI af_err af_dot(af_array *out,
                        const af_array lhs, const af_array rhs,
                        const af_mat_prop optLhs, const af_mat_prop optRhs);

#if AF_API_VERSION >= 35
    /**
        Scalar dot product between two vectors. Also referred to as the inner
        product. Returns the result as a host scalar.

        \param[out] real is the real component of the result of dot operation
        \param[out] imag is the imaginary component of the result of dot operation
        \param[in] lhs The array object on the left hand side
        \param[in] rhs The array object on the right hand side
        \param[in] optLhs Options for lhs. Currently only \ref AF_MAT_NONE and
                   AF_MAT_CONJ are supported.
        \param[in] optRhs Options for rhs. Currently only \ref AF_MAT_NONE and AF_MAT_CONJ are supported

        \return AF_SUCCESS if the process is successful.

        \ingroup blas_func_dot
    */
    AFAPI af_err af_dot_all(double *real, double *imag,
                            const af_array lhs, const af_array rhs,
                            const af_mat_prop optLhs, const af_mat_prop optRhs);
#endif

    /**
        \brief C Interface for transposing a matrix

        \param[out] out the transposed matrix
        \param[in] in an input matrix
        \param[in] conjugate if true, a conjugate transposition is performed

        \return AF_SUCCESS if the process is successful.
        \ingroup blas_func_transpose
    */
    AFAPI af_err af_transpose(af_array *out, af_array in, const bool conjugate);

    /**
        \brief C Interface for transposing a matrix in-place

        \param[in,out] in is the matrix to be transposed in place
        \param[in] conjugate if true, a conjugate transposition is performed

        \ingroup blas_func_transpose
    */
    AFAPI af_err af_transpose_inplace(af_array in, const bool conjugate);


#ifdef __cplusplus
}
#endif
