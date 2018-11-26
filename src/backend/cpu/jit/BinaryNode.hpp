/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <optypes.hpp>
#include <vector>
#include <math.hpp>
#include "Node.hpp"
#include <array>

namespace cpu
{

    template<typename To, typename Ti, af_op_t op>
    struct BinOp
    {
        void eval(jit::array<To> &out,
                  const jit::array<Ti> &lhs,
                  const jit::array<Ti> &rhs,
                  int lim) const
        {
            UNUSED(lhs);
            UNUSED(rhs);
            for (int i = 0; i < lim; i++) {
                out[i] = scalar<To>(0);
            }
        }
    };

namespace jit
{

    template<typename To, typename Ti, af_op_t op>
    class BinaryNode  : public TNode<To>
    {

    protected:
        BinOp<To, Ti, op> m_op;
        TNode<Ti> *m_lhs, *m_rhs;

    public:
        BinaryNode(Node_ptr lhs, Node_ptr rhs) :
            TNode<To>(0, std::max(lhs->getHeight(), rhs->getHeight()) + 1, {{lhs, rhs}}),
            m_lhs(reinterpret_cast<TNode<Ti> *>(lhs.get())),
            m_rhs(reinterpret_cast<TNode<Ti> *>(rhs.get()))
        {
        }

        void calc(int x, int y, int z, int w, int lim) final
        {
            UNUSED(x);
            UNUSED(y);
            UNUSED(z);
            UNUSED(w);
            m_op.eval(this->m_val, m_lhs->m_val, m_rhs->m_val, lim);
        }

        void calc(int idx, int lim) final
        {
            UNUSED(idx);
            m_op.eval(this->m_val, m_lhs->m_val, m_rhs->m_val, lim);
        }
    };

}

}
