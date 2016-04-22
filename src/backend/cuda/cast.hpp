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
#include <err_cuda.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <types.hpp>
#include <JIT/UnaryNode.hpp>
#include <Array.hpp>

namespace cuda
{

template<typename To, typename Ti>
struct CastOp
{
    std::string func;
    CastOp() {
        std::string tmp = std::string("___mk") + afShortName<To>();
        func = cuMangledName<Ti, false>(tmp.c_str());
    }

    const std::string name()
    {
        return func;
    }
};


template<typename To, typename Ti>
struct CastWrapper
{
    Array<To> operator()(const Array<Ti> &in)
    {
        CastOp<To, Ti> cop;
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode *node = new JIT::UnaryNode(irname<To>(),
                                                  afShortName<To>(),
                                                  cop.name(),
                                                  in_node, af_cast_t);
        return createNodeArray<To>(in.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
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
