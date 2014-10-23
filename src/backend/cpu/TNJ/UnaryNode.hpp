#pragma once
#include <af/array.h>
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
        Node *m_child;
        UnOp <To, Ti, op> m_op;
        To m_val;

    public:
        UnaryNode(Node *in) :
            Node(),
            m_child(in),
            m_val(0)
        {
            m_child->addParent(this);
        }

        void *calc(int x, int y, int z, int w)
        {
            if (!m_is_eval) {
                m_val = m_op.eval(*(Ti *)m_child->calc(x, y, z, w));
            }
            return (void *)(&m_val);
        }

        void reset()
        {
            m_is_eval = false;
        }

        void replaceChild(Node *prev, Node *curr)
        {
            if (m_child == prev) m_child = curr;
            else m_child->replaceChild(prev, curr);
        }
    };

}

}
