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

namespace cpu
{
    template<typename To, typename Ti, af_op_t op>
    struct UnOp
    {
        void eval(jit::array<To> &out,
                  const jit::array<Ti> &in, int lim) const
        {
            for (int i = 0; i < lim; i++) {
                out[i] = To(in[i]);
            }
        }
    };

namespace jit
{

    template<typename To, typename Ti, af_op_t op>
    class UnaryNode  : public TNode<To>
    {

    protected:
        UnOp<To, Ti, op> m_op;
        TNode<Ti> *m_child;

    public:
        UnaryNode(Node_ptr child) :
            TNode<To>(0, child->getHeight() + 1, {{child}}),
            m_child(reinterpret_cast<TNode<Ti> *>(child.get()))
        {
        }

        void calc(int x, int y, int z, int w, int lim) final
        {
            m_op.eval(TNode<To>::m_val, m_child->m_val, lim);
        }

        void calc(int idx, int lim) final
        {
            m_op.eval(TNode<To>::m_val, m_child->m_val, lim);
        }

    };

}

}
