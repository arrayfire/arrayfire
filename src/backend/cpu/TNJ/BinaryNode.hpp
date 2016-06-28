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
    struct BinOp
    {
        To eval(Ti lhs, Ti rhs)
        {
            return scalar<To>(0);
        }
    };

namespace TNJ
{

    template<typename To, typename Ti, af_op_t op>
    class BinaryNode  : public Node
    {

    protected:
        Node_ptr m_lhs;
        Node_ptr m_rhs;
        BinOp<To, Ti, op> m_op;
        To m_val;

    public:
        BinaryNode(Node_ptr lhs, Node_ptr rhs) :
            Node(),
            m_lhs(lhs),
            m_rhs(rhs),
            m_val(0)
        {
        }

        void *calc(int x, int y, int z, int w)
        {
            if (calcCurrent(x, y, z, w)) {
                m_val = m_op.eval(*(Ti *)m_lhs->calc(x, y, z, w),
                                  *(Ti *)m_rhs->calc(x, y, z, w));
            }
            return  (void *)&m_val;
        }

        void *calc(int idx)
        {
            if (calcCurrent(idx)) {
                m_val = m_op.eval(*(Ti *)m_lhs->calc(idx),
                                  *(Ti *)m_rhs->calc(idx));
            }
            return (void *)&m_val;
        }

        void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes)
        {
            if (m_is_eval) return;

            m_lhs->getInfo(len, buf_count, bytes);
            m_rhs->getInfo(len, buf_count, bytes);
            len++;

            m_is_eval = true;
            return;
        }

        void reset()
        {
            if (m_is_eval) {
                resetCommonFlags();
                m_lhs->reset();
                m_rhs->reset();
            }
        }

        bool isLinear(const dim_t *dims)
        {
            if (!m_set_is_linear) {
                m_linear = m_lhs->isLinear(dims) && m_rhs->isLinear(dims);
                m_set_is_linear = true;
            }
            return m_linear;
        }
    };

}

}
