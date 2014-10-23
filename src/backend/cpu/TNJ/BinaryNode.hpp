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
        Node *m_lhs;
        Node *m_rhs;
        BinOp<To, Ti, op> m_op;
        To m_val;

    public:
        BinaryNode(Node *lhs, Node *rhs) :
            Node(),
            m_lhs(lhs),
            m_rhs(rhs),
            m_val(0)
        {
            m_lhs->addParent(this);
            m_rhs->addParent(this);
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

        void replaceChild(Node *prev, Node *curr)
        {
            if (m_lhs == prev) m_lhs = curr;
            else m_lhs->replaceChild(prev, curr);

            if (m_rhs == prev) m_rhs = curr;
            else m_rhs->replaceChild(prev, curr);
        }
    };

}

}
