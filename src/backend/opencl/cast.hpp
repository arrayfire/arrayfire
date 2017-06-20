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
#include <complex>
#include <err_opencl.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <JIT/UnaryNode.hpp>
#include <types.hpp>
#include <Array.hpp>

namespace opencl
{

template<typename To, typename Ti>
struct CastOp
{
    const char *name()
    {
        return "";
    }
};

#define CAST_FN(TYPE)                           \
    template<typename Ti>                       \
    struct CastOp<TYPE, Ti>                     \
    {                                           \
        const char *name()                      \
        {                                       \
            return "convert_"#TYPE;             \
        }                                       \
    };

CAST_FN(int)
CAST_FN(uint)
CAST_FN(uchar)
CAST_FN(float)
CAST_FN(double)

#define CAST_CFN(TYPE)                          \
    template<typename Ti>                       \
    struct CastOp<TYPE, Ti>                     \
    {                                           \
        const char *name()                      \
        {                                       \
            return "__convert_"#TYPE;           \
        }                                       \
    };

CAST_CFN(cfloat)
CAST_CFN(cdouble)
CAST_CFN(char)

template<>
struct CastOp<cfloat, cdouble>
{
    const char *name()
    {
        return "__convert_z2c";
    }
};

template<>
struct CastOp<cdouble, cfloat>
{
    const char *name()
    {
        return "__convert_c2z";
    }
};

template<>
struct CastOp<cfloat, cfloat>
{
    const char *name()
    {
        return "__convert_c2c";
    }
};

template<>
struct CastOp<cdouble, cdouble>
{
    const char *name()
    {
        return "__convert_z2z";
    }
};

#undef CAST_FN
#undef CAST_CFN

template<typename To, typename Ti>
struct CastWrapper
{
    Array<To> operator()(const Array<Ti> &in)
    {
        CastOp<To, Ti> cop;
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode *node = new JIT::UnaryNode(dtype_traits<To>::getName(),
                                                  shortname<To>(true),
                                                  cop.name(),
                                                  in_node, af_cast_t);
        return createNodeArray<To>(in.dims(), JIT::Node_ptr(node));
    }
};

template<typename T>
struct CastWrapper<T, T>
{
    Array<T> operator()(const Array<T> &in)
    {
        return in;
    }
};

template<typename To, typename Ti>
Array<To> cast(const Array<Ti> &in)
{
    CastWrapper<To, Ti> cast_op;
    return cast_op(in);
}

}
