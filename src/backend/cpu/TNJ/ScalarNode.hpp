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

        void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes)
        {
            if (m_is_eval) return;
            len++;
            m_is_eval = true;
            return;
        }

        void reset(bool reset_off=true) { m_is_eval = false; }
    };
}

}
