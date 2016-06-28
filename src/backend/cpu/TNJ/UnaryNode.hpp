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
    class UnaryNode  : public Node
    {

    protected:
        Node_ptr m_child;
        UnOp <To, Ti, op> m_op;
        To m_val;

    public:
        UnaryNode(Node_ptr in) :
            Node(),
            m_child(in),
            m_val(0)
        {
        }

        void *calc(int x, int y, int z, int w)
        {
            if (calcCurrent(x, y, z, w)) {
                m_val = m_op.eval(*(Ti *)m_child->calc(x, y, z, w));
            }
            return (void *)(&m_val);
        }

        void *calc(int idx)
        {
            if (calcCurrent(idx)) {
                m_val = m_op.eval(*(Ti *)m_child->calc(idx));
            }
            return (void *)&m_val;
        }

        void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes)
        {
            if (m_is_eval) return;

            m_child->getInfo(len, buf_count, bytes);
            len++;

            m_is_eval = true;
            return;
        }

        void reset()
        {
            if (m_is_eval) {
                resetCommonFlags();
                m_child->reset();
            }
        }

        bool isLinear(const dim_t *dims)
        {
            if (!m_set_is_linear) {
                m_linear = m_child->isLinear(dims);
                m_set_is_linear = true;
            }
            return m_linear;
        }
    };

}

}
