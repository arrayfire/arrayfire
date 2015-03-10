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

/// \defgroup blas_funcs BLAS Functions
/// @{

#include <af/array.h>
#include "af/defines.h"

#ifdef __cplusplus
extern "C" {
#endif
    typedef enum af_transpose_enum {
        AF_NO_TRANSPOSE,
        AF_TRANSPOSE,
        AF_CONJUGATE_TRANSPOSE
    } af_blas_transpose;
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace af
{
    /**
        \brief Matrix multiply on two arrays

        \copydetails blas_func_matmul

        \param lhs The array object on the left hand side
        \param rhs The array object on the right hand side
        \param optLhs Transpose operation before the function is performed
        \param optRhs Transpose operation before the function is performed
        \returns The result of the matrix multiplication of lhs, rhs

        \ingroup blas_func_matmul
     */
    AFAPI array matmul(const array &lhs, const array &rhs,
                       af_blas_transpose optLhs = AF_NO_TRANSPOSE,
                       af_blas_transpose optRhs = AF_NO_TRANSPOSE);

    /**
       \brief Matrix multiply on two arrays

       \copydetails blas_func_matmul

       \param lhs The array object on the left hand side
       \param rhs The array object on the right hand side
       \returns The result of the matrix multiplication of lhs, transpose(rhs)

       \ingroup blas_func_matmulnt
    */
    AFAPI array matmulNT(const array &lhs, const array &rhs);

    /**
       \brief Matrix multiply on two arrays

       \copydetails blas_func_matmul

       \param lhs The array object on the left hand side
       \param rhs The array object on the right hand side
       \returns The result of the matrix multiplication of transpose(lhs), rhs

       \ingroup blas_func_matmultn
    */
    AFAPI array matmulTN(const array &lhs, const array &rhs);

    /**
       \brief Matrix multiply on two arrays

       \copydetails blas_func_matmul

       \param lhs The array object on the left hand side
       \param rhs The array object on the right hand side
       \returns The result of the matrix multiplication of transpose(lhs), transpose(rhs)

       \ingroup blas_func_matmultt
    */
    AFAPI array matmulTT(const array &lhs, const array &rhs);

    /**
        \brief Dot Product

        Scalar dot product between two vectors.  Also referred to as the inner
        product.

        \democode{
        // compute scalar dot product
        array x = randu(100), y = randu(100);
        print(dot<float>(x,y));
        }
        \ingroup blas_func_dot
    */
    AFAPI array dot   (const array &lhs, const array &rhs,
                       af_blas_transpose optLhs = AF_NO_TRANSPOSE,
                       af_blas_transpose optRhs = AF_NO_TRANSPOSE);

    /**
        \brief Transposes a matrix

        \copydetails blas_func_transpose

        \param in Input Matrix
        \param conjugate If true a congugate transposition is performed
        \returns Transposed matrix
        \ingroup blas_func_transpose
    */
    AFAPI array transpose(const array& in, const bool conjugate = false);
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

        \param out Pointer to the output \ref af_array
        \param lhs A 2D matrix \ref af_array object
        \param rhs A 2D matrix \ref af_array object
        \param optLhs Transpose operation before the function is performed
        \param optRhs Transpose operation before the function is performed

        \returns AF_SUCCESS if the process is successful.
        \ingroup matmul_ptr blas_func_matmul
     */
    AFAPI af_err af_matmul( af_array *out ,
                            const af_array lhs, const af_array rhs,
                            af_blas_transpose optLhs, af_blas_transpose optRhs);


    /**
        Scalar dot product between two vectors.  Also referred to as the inner
        product.

        \democode{
        // compute scalar dot product
        array x = randu(100), y = randu(100);
        print(dot<float>(x,y));
        }
        \ingroup matmul_ptr blas_func_dot
    */

    AFAPI af_err af_dot(    af_array *out,
                            const af_array lhs, const af_array rhs,
                            af_blas_transpose optLhs, af_blas_transpose optRhs);

    /**
        \brief Transposes a matrix

        This funciton will tranpose the matrix in.

        \param out The transposed matrix
        \param in Input matrix which will be transposed
        \param conjugate Perform a congugate transposition
        \returns Transposed matrix
        \ingroup matmul_mat blas_func_transpose
    */
    AFAPI af_err af_transpose(af_array *out, af_array in, const bool conjugate);


#ifdef __cplusplus
}
#endif
///
/// @}
///
