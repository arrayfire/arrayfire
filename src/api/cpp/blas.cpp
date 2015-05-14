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
                 const matProp optLhs, const matProp optRhs)
    {
        af_array out = 0;
        AF_THROW(af_matmul(&out, lhs.get(), rhs.get(), optLhs, optRhs));
        return array(out);
    }

    array matmulNT(const array &lhs, const array &rhs)
    {
        af_array out = 0;
        AF_THROW(af_matmul(&out, lhs.get(), rhs.get(),
                           AF_MAT_NONE, AF_MAT_TRANS));
        return array(out);
    }

    array matmulTN(const array &lhs, const array &rhs)
    {
        af_array out = 0;
        AF_THROW(af_matmul(&out, lhs.get(), rhs.get(),
                           AF_MAT_TRANS, AF_MAT_NONE));
        return array(out);
    }

    array matmulTT(const array &lhs, const array &rhs)
    {
        af_array out = 0;
        AF_THROW(af_matmul(&out, lhs.get(), rhs.get(),
                           AF_MAT_TRANS, AF_MAT_TRANS));
        return array(out);
    }

    array dot   (const array &lhs, const array &rhs,
                 const matProp optLhs, const matProp optRhs)
    {
        af_array out = 0;
        AF_THROW(af_dot(&out, lhs.get(), rhs.get(), optLhs, optRhs));
        return array(out);
    }
}
