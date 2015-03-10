/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/blas.h>
#include "error.hpp"

namespace af
{
    array matmul(const array &lhs, const array &rhs,
                 af_blas_transpose optLhs, af_blas_transpose optRhs)
    {
        af_array out = 0;
        AF_THROW(af_matmul(&out, lhs.get(), rhs.get(), optLhs, optRhs));
        return array(out);
    }

    array matmulNT(const array &lhs, const array &rhs)
    {
        af_array out = 0;
        AF_THROW(af_matmul(&out, lhs.get(), rhs.get(),
                           AF_NO_TRANSPOSE, AF_TRANSPOSE));
        return array(out);
    }

    array matmulTN(const array &lhs, const array &rhs)
    {
        af_array out = 0;
        AF_THROW(af_matmul(&out, lhs.get(), rhs.get(),
                           AF_TRANSPOSE, AF_NO_TRANSPOSE));
        return array(out);
    }

    array matmulTT(const array &lhs, const array &rhs)
    {
        af_array out = 0;
        AF_THROW(af_matmul(&out, lhs.get(), rhs.get(),
                           AF_TRANSPOSE, AF_TRANSPOSE));
        return array(out);
    }

    array dot   (const array &lhs, const array &rhs,
                 af_blas_transpose optLhs, af_blas_transpose optRhs)
    {
        af_array out = 0;
        AF_THROW(af_dot(&out, lhs.get(), rhs.get(), optLhs, optRhs));
        return array(out);
    }
}
