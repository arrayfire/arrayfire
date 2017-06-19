/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <optypes.hpp>
#include <binary.hpp>
#include <JIT/UnaryNode.hpp>

namespace cuda
{
    template<typename To, typename Ti>
    Array<To> cplx(const Array<Ti> &lhs, const Array<Ti> &rhs, const af::dim4 &odims)
    {
        return createBinaryNode<To, Ti, af_cplx2_t>(lhs, rhs, odims);
    }

    template<typename To, typename Ti>
    Array<To> real(const Array<Ti> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode *node = new JIT::UnaryNode(getFullName<To>(),
                                                  shortname<To>(true),
                                                  "__creal",
                                                  in_node, af_real_t);

        return createNodeArray<To>(in.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }

    template<typename To, typename Ti>
    Array<To> imag(const Array<Ti> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode *node = new JIT::UnaryNode(getFullName<To>(),
                                                  shortname<To>(true),
                                                  "__cimag",
                                                  in_node, af_imag_t);

        return createNodeArray<To>(in.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }

    template<typename T> static const char *abs_name() { return "fabs"; }
    template<> STATIC_ const char *abs_name<cfloat>() { return "__cabsf"; }
    template<> STATIC_ const char *abs_name<cdouble>() { return "__cabs"; }

    template<typename To, typename Ti>
    Array<To> abs(const Array<Ti> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode *node = new JIT::UnaryNode(getFullName<To>(),
                                                  shortname<To>(true),
                                                  abs_name<Ti>(),
                                                  in_node, af_abs_t);

        return createNodeArray<To>(in.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }

    template<typename T> static const char *conj_name() { return "__noop"; }
    template<> STATIC_ const char *conj_name<cfloat>() { return "__cconjf"; }
    template<> STATIC_ const char *conj_name<cdouble>() { return "__cconj"; }

    template<typename T>
    Array<T> conj(const Array<T> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode *node = new JIT::UnaryNode(getFullName<T>(),
                                                  shortname<T>(true),
                                                  conj_name<T>(),
                                                  in_node, af_conj_t);

        return createNodeArray<T>(in.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }
}
