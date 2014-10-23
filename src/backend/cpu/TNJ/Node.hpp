#pragma once
#include <af/array.h>
#include <optypes.hpp>
#include <vector>
#include "Node.hpp"

namespace cpu
{

namespace TNJ
{

    class Node
    {

    protected:
        bool m_is_eval;
        std::vector<Node *> m_parents;

    public:
        Node() : m_is_eval(false) {}

        virtual void *calc(int x, int y, int z, int w)
        {
            m_is_eval = true;
            return NULL;
        }

        virtual void reset() { m_is_eval = false;}
        virtual void replaceChild(Node *prev, Node *curr) {};

        void addParent(Node *node)
        {
            m_parents.push_back(node);
        }

        void replace(Node *node)
        {
            for (size_t i = 0; i < m_parents.size(); i++) {
                m_parents[i]->replaceChild(this, node);
            }
        }

        virtual ~Node() {}
    };
}

}
