/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <complex>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <optypes.hpp>
#include <err_cpu.hpp>
#include <JIT/BinaryNode.hpp>
#include <JIT/UnaryNode.hpp>

namespace cpu
{

    template<typename To, typename Ti>
    struct BinOp<To, Ti, af_cplx2_t>
    {
        void eval(JIT::array<To> &out,
                  const JIT::array<Ti> &lhs,
                  const JIT::array<Ti> &rhs,
                  int lim)
        {
            for (int i = 0; i < lim; i++) {
                out[i] = To(lhs[i], rhs[i]);
            }
        }
    };

    template<typename To, typename Ti>
    Array<To> cplx(const Array<Ti> &lhs, const Array<Ti> &rhs, const af::dim4 &odims)
    {
        JIT::Node_ptr lhs_node = lhs.getNode();
        JIT::Node_ptr rhs_node = rhs.getNode();

        JIT::BinaryNode<To, Ti, af_cplx2_t> *node =
            new JIT::BinaryNode<To, Ti, af_cplx2_t>(lhs_node, rhs_node);

        return createNodeArray<To>(odims,
                                   JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }

#define CPLX_UNARY_FN(op)                               \
    template<typename To, typename Ti>                  \
    struct UnOp<To, Ti, af_##op##_t>                    \
    {                                                   \
        void eval(JIT::array<To> &out,                  \
                  const JIT::array<Ti> &in, int lim)    \
        {                                               \
            for (int i = 0; i < lim; i++) {             \
                out[i] = std::op(in[i]);                \
            }                                           \
        }                                               \
    };                                                  \

    CPLX_UNARY_FN(real)
    CPLX_UNARY_FN(imag)
    CPLX_UNARY_FN(conj)
    CPLX_UNARY_FN(abs)

    template<typename To, typename Ti>
    Array<To> real(const Array<Ti> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode<To, Ti, af_real_t> *node = new JIT::UnaryNode<To, Ti, af_real_t>(in_node);

        return createNodeArray<To>(in.dims(),
                                   JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }

    template<typename To, typename Ti>
    Array<To> imag(const Array<Ti> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode<To, Ti, af_imag_t> *node = new JIT::UnaryNode<To, Ti, af_imag_t>(in_node);

        return createNodeArray<To>(in.dims(),
                                   JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }

    template<typename To, typename Ti>
    Array<To> abs(const Array<Ti> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode<To, Ti, af_abs_t> *node = new JIT::UnaryNode<To, Ti, af_abs_t>(in_node);

        return createNodeArray<To>(in.dims(),
                                   JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }

    template<typename T>
    Array<T> conj(const Array<T> &in)
    {
        JIT::Node_ptr in_node = in.getNode();
        JIT::UnaryNode<T, T, af_conj_t> *node = new JIT::UnaryNode<T, T, af_conj_t>(in_node);

        return createNodeArray<T>(in.dims(),
                                  JIT::Node_ptr(reinterpret_cast<JIT::Node *>(node)));
    }
}
