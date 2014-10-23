#pragma once
#include <af/array.h>
#include <optypes.hpp>
#include <vector>
#include "Node.hpp"

namespace cpu
{

namespace TNJ
{

    template<typename T>
    class ScalarNode : public Node
    {

    protected:
        T m_val;

    public:
        ScalarNode(T val) : Node(), m_val(val) {}

        void *calc(int x, int y, int z, int w)
        {
            return (void *)(&m_val);
        }

        void reset() {}
    };
}

}
