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


    /**
        \brief Dot Product

        Scalar dot product between two vectors.  Also referred to as the inner
        product.

        \code
        // compute scalar dot product
        array x = randu(100), y = randu(100);
        af_print(dot(x,y));
        \endcode

        \param[in] lhs The array object on the left hand side
        \param[in] rhs The array object on the right hand side
        \param[in] optLhs Options for lhs. Currently only \ref AF_MAT_NONE and
                   AF_MAT_CONJ are supported.
        \param[in] optRhs Options for rhs. Currently only \ref AF_MAT_NONE and AF_MAT_CONJ are supported
        \return The result of the dot product of lhs, rhs

        \note optLhs and optRhs can only be one of \ref AF_MAT_NONE or \ref AF_MAT_CONJ
        \note optLhs = AF_MAT_CONJ and optRhs = AF_MAT_NONE will run conjugate dot operation.
        \note This function is not supported in GFOR

        \returns out = dot(lhs, rhs)

        \ingroup blas_func_dot
    */
    AFAPI array dot   (const array &lhs, const array &rhs,
                       const matProp optLhs = AF_MAT_NONE,
                       const matProp optRhs = AF_MAT_NONE);

#if AF_API_VERSION >= 35
    /**
        \brief Return the dot product of two vectors as a scalar

        Scalar dot product between two vectors. Also referred to as the inner
        product.

        \code
        // compute scalar dot product
        array x = randu(100), y = randu(100);
        float h_dot = dot<float>(x,y);
        \endcode

        \param[in] lhs The array object on the left hand side
        \param[in] rhs The array object on the right hand side
        \param[in] optLhs Options for lhs. Currently only \ref AF_MAT_NONE and
                   AF_MAT_CONJ are supported.
        \param[in] optRhs Options for rhs. Currently only \ref AF_MAT_NONE and AF_MAT_CONJ are supported
        \return The result of the dot product of lhs, rhs as a host scalar

        \note optLhs and optRhs can only be one of \ref AF_MAT_NONE or \ref AF_MAT_CONJ
        \note optLhs = AF_MAT_CONJ and optRhs = AF_MAT_NONE will run conjugate dot operation.
        \note This function is not supported in GFOR

        \returns out = dot(lhs, rhs)

        \ingroup blas_func_dot
    */
    template<typename T> T dot(const array &lhs, const array &rhs,
                               const matProp optLhs = AF_MAT_NONE,
                               const matProp optRhs = AF_MAT_NONE);
#endif

    /**
        \brief Transposes a matrix

        \copydetails blas_func_transpose

        \param[in] in Input Matrix
        \param[in] conjugate If true a congugate transposition is performed
        \return Transposed matrix
        \ingroup blas_func_transpose
    */
    AFAPI array transpose(const array& in, const bool conjugate = false);

    /**
        \brief Transposes a matrix in-place

        \copydetails blas_func_transpose

        \param[in,out] in is the matrix to be transposed in place
        \param[in] conjugate If true a congugate transposition is performed

        \ingroup blas_func_transpose
    */
    AFAPI void transposeInPlace(array& in, const bool conjugate = false);
}
#endif

#ifdef __cplusplus
extern "C" {
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
        \brief Transposes a matrix

        This funciton will tranpose the matrix in.

        \param[out] out The transposed matrix
        \param[in] in Input matrix which will be transposed
        \param[in] conjugate Perform a congugate transposition

        \return AF_SUCCESS if the process is successful.
        \ingroup blas_func_transpose
    */
    AFAPI af_err af_transpose(af_array *out, af_array in, const bool conjugate);

    /**
        \brief Transposes a matrix in-place

        \copydetails blas_func_transpose

        \param[in,out] in is the matrix to be transposed in place
        \param[in] conjugate If true a congugate transposition is performed

        \ingroup blas_func_transpose
    */
    AFAPI af_err af_transpose_inplace(af_array in, const bool conjugate);


#ifdef __cplusplus
}
#endif
