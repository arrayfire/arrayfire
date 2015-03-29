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
#include <af/dim4.hpp>
#include <Array.hpp>
#include <optypes.hpp>
#include <err_cuda.hpp>
#include <JIT/BinaryNode.hpp>
#include <JIT/UnaryNode.hpp>

namespace cuda
{
    template<typename T> static const std::string cplx_name() { return "@___noop"; }
    template<> STATIC_ const std::string cplx_name<cfloat >() { return cuMangledName<float , true>("___cplx"); }
    template<> STATIC_ const std::string cplx_name<cdouble>() { return cuMangledName<double, true>("___cplx"); }

    template<typename T> static const std::string real_name() { return "@___noop"; }
    template<> STATIC_ const std::string real_name<cfloat >() { return cuMangledName<cfloat , false>("___real"); }
    template<> STATIC_ const std::string real_name<cdouble>() { return cuMangledName<cdouble, false>("___real"); }

    template<typename T> static const std::string imag_name() { return "@___noop"; }
    template<> STATIC_ const std::string imag_name<cfloat >() { return cuMangledName<cfloat , false>("___imag"); }
    template<> STATIC_ const std::string imag_name<cdouble>() { return cuMangledName<cdouble, false>("___imag"); }

    template<typename T> static const std::string abs_name() { return "@___noop"; }
    template<> STATIC_ const std::string abs_name<float  >() { return cuMangledName<float  , false>("___abs"); }
    template<> STATIC_ const std::string abs_name<double >() { return cuMangledName<double , false>("___abs"); }
    template<> STATIC_ const std::string abs_name<cfloat >() { return cuMangledName<cfloat , false>("___abs"); }
    template<> STATIC_ const std::string abs_name<cdouble>() { return cuMangledName<cdouble, false>("___abs"); }

    template<typename T> static const std::string conj_name() { return "@___noop"; }
    template<> STATIC_ const std::string conj_name<cfloat >() { return cuMangledName<cfloat , false>("___conj"); }
    template<> STATIC_ const std::string conj_name<cdouble>() { return cuMangledName<cdouble, false>("___conj"); }

    template<typename To, typename Ti>
    Array<To> cplx(const Array<Ti> &lhs, const Array<Ti> &rhs, const af::dim4 &odims)
    {
        JIT::Node_ptr lhs_node = lhs.getNode();
        JIT::Node_ptr rhs_node = rhs.getNode();

        JIT::BinaryNode *node = new JIT::BinaryNode(irname<To>(),
                                                    afShortName<To>(),
                                                    cplx_name<To>(),
                                                    lhs_node,
                                                    rhs_node, (int)(af_cplx2_t));

        return createNodeArray<To>(odims, JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }

    template<typename To, typename Ti>
    Array<To> real(const Array<Ti> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode *node = new JIT::UnaryNode(irname<To>(),
                                                  afShortName<To>(),
                                                  real_name<Ti>(),
                                                  in_node, af_real_t);

        return createNodeArray<To>(in.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }

    template<typename To, typename Ti>
    Array<To> imag(const Array<Ti> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode *node = new JIT::UnaryNode(irname<To>(),
                                                  afShortName<To>(),
                                                  imag_name<Ti>(),
                                                  in_node, af_imag_t);

        return createNodeArray<To>(in.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }

    template<typename To, typename Ti>
    Array<To> abs(const Array<Ti> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode *node = new JIT::UnaryNode(irname<To>(),
                                                  afShortName<To>(),
                                                  abs_name<Ti>(),
                                                  in_node, af_abs_t);

        return createNodeArray<To>(in.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }

    template<typename T>
    Array<T> conj(const Array<T> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode *node = new JIT::UnaryNode(irname<T>(),
                                                  afShortName<T>(),
                                                  conj_name<T>(),
                                                  in_node, af_conj_t);

        return createNodeArray<T>(in.dims(), JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }
}
