/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/array.h>
#include <optypes.hpp>
#include <vector>
#include <math.hpp>

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
            if (!m_is_eval) {
                m_val = m_op.eval(*(Ti *)m_lhs->calc(x, y, z, w),
                                  *(Ti *)m_rhs->calc(x, y, z, w));
            }
            return  (void *)&m_val;
        }

        void reset()
        {
            m_is_eval = false;
        }
    };

}

}
