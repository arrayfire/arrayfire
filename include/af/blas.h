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

#include "af/defines.h"

#ifdef __cplusplus
extern "C" {
#endif
    /**
       \ingroup blas_func_matmul
    */
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace af
{
    class array;
    /**
        \brief Matrix multiply on two arrays

        \copydetails blas_func_matmul

        \param[in] lhs The array object on the left hand side
        \param[in] rhs The array object on the right hand side
        \param[in] optLhs Transpose operation before the function is performed
        \param[in] optRhs Transpose operation before the function is performed
        \return The result of the matrix multiplication of lhs, rhs

        \ingroup blas_func_matmul
     */
    AFAPI array matmul(const array &lhs, const array &rhs,
                       const af::trans optLhs = AF_NO_TRANS,
                       const af::trans optRhs = AF_NO_TRANS);

    /**
       \brief Matrix multiply on two arrays

       \copydetails blas_func_matmul

       \param[in] lhs The array object on the left hand side
       \param[in] rhs The array object on the right hand side
       \return The result of the matrix multiplication of \p lhs, transpose(\p rhs)

       \ingroup blas_func_matmul
    */
    AFAPI array matmulNT(const array &lhs, const array &rhs);

    /**
       \brief Matrix multiply on two arrays

       \copydetails blas_func_matmul

       \param[in] lhs The array object on the left hand side
       \param[in] rhs The array object on the right hand side
       \return The result of the matrix multiplication of transpose(\p lhs), \p rhs

       \ingroup blas_func_matmul
    */
    AFAPI array matmulTN(const array &lhs, const array &rhs);

    /**
       \brief Matrix multiply on two arrays

       \copydetails blas_func_matmul

       \param[in] lhs The array object on the left hand side
       \param[in] rhs The array object on the right hand side
       \return The result of the matrix multiplication of transpose(\p lhs), transpose(\p rhs)

       \ingroup blas_func_matmul
    */
    AFAPI array matmulTT(const array &lhs, const array &rhs);

    /**
        \brief Dot Product

        Scalar dot product between two vectors.  Also referred to as the inner
        product.

        \democode{
        // compute scalar dot product
        array x = randu(100), y = randu(100);
        af_print(dot(x,y));
        }
        \ingroup blas_func_dot
    */
    AFAPI array dot   (const array &lhs, const array &rhs,
                       const af::trans optLhs = AF_NO_TRANS,
                       const af::trans optRhs = AF_NO_TRANS);

    /**
        \brief Transposes a matrix

        \copydetails blas_func_transpose

        \param[in] in Input Matrix
        \param[in] conjugate If true a congugate transposition is performed
        \return Transposed matrix
        \ingroup blas_func_transpose
    */
    AFAPI array transpose(const array& in, const bool conjugate = false);

    AFAPI void transposeInPlace(array& in, const bool conjugate = false);
    /**
      }@
    */
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
        \brief Matrix multiply on two \ref af_array

        \details Performs a matrix multiplication on two arrays (lhs, rhs).

        \param[out] out Pointer to the output \ref af_array
        \param[in] lhs A 2D matrix \ref af_array object
        \param[in] rhs A 2D matrix \ref af_array object
        \param[in] optLhs Transpose operation before the function is performed
        \param[in] optRhs Transpose operation before the function is performed

        \return AF_SUCCESS if the process is successful.
        \ingroup blas_func_matmul
     */
    AFAPI af_err af_matmul( af_array *out ,
                            const af_array lhs, const af_array rhs,
                            const af_transpose_t optLhs, const af_transpose_t optRhs);


    /**
        Scalar dot product between two vectors.  Also referred to as the inner
        product.

        \democode{
        // compute scalar dot product
        array x = randu(100), y = randu(100);
        print(dot<float>(x,y));
        }
        \ingroup blas_func_dot
    */

    AFAPI af_err af_dot(    af_array *out,
                            const af_array lhs, const af_array rhs,
                            const af_transpose_t optLhs, const af_transpose_t optRhs);

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

    AFAPI af_err af_transpose_inplace(af_array in, const bool conjugate);


#ifdef __cplusplus
}
#endif
