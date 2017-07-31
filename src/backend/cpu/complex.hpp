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
#include <TNJ/BinaryNode.hpp>
#include <TNJ/UnaryNode.hpp>

namespace cpu
{

    template<typename To, typename Ti>
    struct BinOp<To, Ti, af_cplx2_t>
    {
        void eval(TNJ::array<To> &out,
                  const TNJ::array<Ti> &lhs,
                  const TNJ::array<Ti> &rhs,
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
        TNJ::Node_ptr lhs_node = lhs.getNode();
        TNJ::Node_ptr rhs_node = rhs.getNode();

        TNJ::BinaryNode<To, Ti, af_cplx2_t> *node =
            new TNJ::BinaryNode<To, Ti, af_cplx2_t>(lhs_node, rhs_node);

        return createNodeArray<To>(odims, TNJ::Node_ptr(
                                       reinterpret_cast<TNJ::Node *>(node)));
    }

#define CPLX_UNARY_FN(op)                               \
    template<typename To, typename Ti>                  \
    struct UnOp<To, Ti, af_##op##_t>                    \
    {                                                   \
        void eval(TNJ::array<To> &out,                  \
                  const TNJ::array<Ti> &in, int lim)    \
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
        TNJ::Node_ptr in_node = in.getNode();
        TNJ::UnaryNode<To, Ti, af_real_t> *node = new TNJ::UnaryNode<To, Ti, af_real_t>(in_node);

        return createNodeArray<To>(in.dims(),
                                   TNJ::Node_ptr(reinterpret_cast<TNJ::Node *>(node)));
    }

    template<typename To, typename Ti>
    Array<To> imag(const Array<Ti> &in)
    {
        TNJ::Node_ptr in_node = in.getNode();
        TNJ::UnaryNode<To, Ti, af_imag_t> *node = new TNJ::UnaryNode<To, Ti, af_imag_t>(in_node);

        return createNodeArray<To>(in.dims(),
                                   TNJ::Node_ptr(reinterpret_cast<TNJ::Node *>(node)));
    }

    template<typename To, typename Ti>
    Array<To> abs(const Array<Ti> &in)
    {
        TNJ::Node_ptr in_node = in.getNode();
        TNJ::UnaryNode<To, Ti, af_abs_t> *node = new TNJ::UnaryNode<To, Ti, af_abs_t>(in_node);

        return createNodeArray<To>(in.dims(),
                                   TNJ::Node_ptr(reinterpret_cast<TNJ::Node *>(node)));
    }

    template<typename T>
    Array<T> conj(const Array<T> &in)
    {
        TNJ::Node_ptr in_node = in.getNode();
        TNJ::UnaryNode<T, T, af_conj_t> *node = new TNJ::UnaryNode<T, T, af_conj_t>(in_node);

        return createNodeArray<T>(in.dims(),
                                  TNJ::Node_ptr(reinterpret_cast<TNJ::Node *>(node)));
    }
}
