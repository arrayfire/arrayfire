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
#include <err_cuda.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <types.hpp>
#include <JIT/UnaryNode.hpp>

namespace cuda
{

template<typename To, typename Ti>
struct CastOp
{
    std::string to, from, func;
    CastOp(): to(shortname<To>(true)), from(shortname<Ti>(false)) {
        func = std::string("@___mk") + to + from;
    }
    const char *name()
    {
        return func.c_str();
    }
};

template<typename To, typename Ti>
Array<To>* cast(const Array<Ti> &in)
{
    CastOp<To, Ti> cop;
    JIT::Node_ptr in_node = in.getNode();

    JIT::UnaryNode *node = new JIT::UnaryNode(irname<To>(),
                                              cop.name(),
                                              in_node, af_cast_t);

    return createNodeArray<To>(in.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
}

}
