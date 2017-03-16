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
        To eval(Ti in)
        {
            return scalar<To>(0);
        }
    };

namespace TNJ
{

    template<typename To, typename Ti, af_op_t op>
    class UnaryNode  : public TNode<To>
    {

    protected:
        UnOp <To, Ti, op> m_op;
        TNode<Ti> *m_child;

    public:
        UnaryNode(Node_ptr child) :
            TNode<To>(0, child->getHeight() + 1, {child}),
            m_child(reinterpret_cast<TNode<Ti> *>(child.get()))
        {
        }


        void calc(int x, int y, int z, int w)
        {
            this->m_val = m_op.eval(m_child->m_val);
        }

        void calc(int idx)
        {
            this->m_val = m_op.eval(m_child->m_val);
        }

    };

}

}
