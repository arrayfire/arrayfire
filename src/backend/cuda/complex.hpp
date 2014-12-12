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
#include <JIT/UnaryNode.hpp>

namespace cuda
{
    template<typename T> static const char *cplx_name() { return "@___noop"; }
	template<> STATIC_ const char *cplx_name<cfloat>() { return "@___cplxss"; }
	template<> STATIC_ const char *cplx_name<cdouble>() { return "@___cplxdd"; }

    template<typename T> static const char *real_name() { return "@___noop"; }
    template<> STATIC_ const char *real_name<cfloat>() { return "@___realc"; }
    template<> STATIC_ const char *real_name<cdouble>() { return "@___realz"; }

    template<typename T> static const char *imag_name() { return "@___noop"; }
    template<> STATIC_ const char *imag_name<cfloat>() { return "@___imagc"; }
    template<> STATIC_ const char *imag_name<cdouble>() { return "@___imagz"; }

    template<typename T> static const char *abs_name() { return "@___noop"; }
    template<> STATIC_ const char *abs_name<float>() { return "@___abss"; }
    template<> STATIC_ const char *abs_name<double>() { return "@___absd"; }
    template<> STATIC_ const char *abs_name<cfloat>() { return "@___absc"; }
    template<> STATIC_ const char *abs_name<cdouble>() { return "@___absz"; }

    template<typename T> static const char *conj_name() { return "@___noop"; }
    template<> STATIC_ const char *conj_name<cfloat>() { return "@___conjc"; }
    template<> STATIC_ const char *conj_name<cdouble>() { return "@___conjz"; }

    template<typename To, typename Ti>
    Array<To>* cplx(const Array<Ti> &lhs, const Array<Ti> &rhs)
    {
        JIT::Node_ptr lhs_node = lhs.getNode();
        JIT::Node_ptr rhs_node = rhs.getNode();

        JIT::BinaryNode *node = new JIT::BinaryNode(irname<To>(),
                                                    cplx_name<To>(),
                                                    lhs_node,
                                                    rhs_node, (int)(af_cplx2_t));

        return createNodeArray<To>(lhs.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }

    template<typename To, typename Ti>
    Array<To>* real(const Array<Ti> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode *node = new JIT::UnaryNode(irname<To>(),
                                                  real_name<Ti>(),
                                                  in_node, af_real_t);

        return createNodeArray<To>(in.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }

    template<typename To, typename Ti>
    Array<To>* imag(const Array<Ti> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode *node = new JIT::UnaryNode(irname<To>(),
                                                  imag_name<Ti>(),
                                                  in_node, af_imag_t);

        return createNodeArray<To>(in.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }

    template<typename To, typename Ti>
    Array<To>* abs(const Array<Ti> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode *node = new JIT::UnaryNode(irname<To>(),
                                                  abs_name<Ti>(),
                                                  in_node, af_abs_t);

        return createNodeArray<To>(in.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }

    template<typename T>
    Array<T>* conj(const Array<T> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode *node = new JIT::UnaryNode(irname<T>(),
                                                  conj_name<T>(),
                                                  in_node, af_conj_t);

        return createNodeArray<T>(in.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }
}
