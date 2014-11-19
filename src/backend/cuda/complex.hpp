/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/array.h>
#include <Array.hpp>
#include <optypes.hpp>
#include <err_cuda.hpp>
#include <JIT/BinaryNode.hpp>

namespace cuda
{
    template<typename T> static const char *cplx2_name() { return "___noop"; }
	template<> STATIC_ const char *cplx2_name<cfloat>() { return "___cplxCss"; }
	template<> STATIC_ const char *cplx2_name<cdouble>() { return "___cplxZdd"; }

    template<typename To, typename Ti>
    Array<To>* complexOp(const Array<Ti> &lhs, const Array<Ti> &rhs)
    {
        JIT::Node_ptr lhs_node = lhs.getNode();
        JIT::Node_ptr rhs_node = rhs.getNode();

        JIT::BinaryNode *node = new JIT::BinaryNode(irname<To>(),
                                                    cplx2_name<To>(),
                                                    lhs_node,
                                                    rhs_node, (int)(af_cplx2_t));

        return createNodeArray<To>(lhs.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }
}
