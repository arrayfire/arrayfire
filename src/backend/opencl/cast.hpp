/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <complex>
#include <err_opencl.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <JIT/UnaryNode.hpp>
#include <types.hpp>

namespace opencl
{

template<typename To>
struct CastOp
{
    const char *name()
    {
        return "";
    }
};

#define CAST_FN(TYPE)                           \
    template<>                                  \
    struct CastOp<TYPE>                         \
    {                                           \
        const char *name()                      \
        {                                       \
            return "convert_"#TYPE;             \
        }                                       \
    };

CAST_FN(int)
CAST_FN(uint)
CAST_FN(char)
CAST_FN(uchar)
CAST_FN(float)
CAST_FN(double)

#define CAST_CFN(TYPE)                          \
    template<>                                  \
    struct CastOp<TYPE>                         \
    {                                           \
        const char *name()                      \
        {                                       \
            return "__convert_"#TYPE;           \
        }                                       \
    };


CAST_CFN(cfloat)
CAST_CFN(cdouble)

#undef CAST_FN
#undef CAST_CFN

template<typename To, typename Ti>
Array<To>* cast(const Array<Ti> &in)
{
    CastOp<To> cop;
    JIT::Node *in_node = in.getNode();

    JIT::UnaryNode *node = new JIT::UnaryNode(dtype_traits<To>::getName(),
                                              cop.name(),
                                              in_node, af_cast_t);

    return createNodeArray<To>(in.dims(), reinterpret_cast<JIT::Node *>(node));
}

}
