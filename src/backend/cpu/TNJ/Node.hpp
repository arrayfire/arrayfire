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
#include <memory>

namespace cpu
{

namespace TNJ
{

    class Node
    {

    protected:
        bool m_is_eval;

    public:
        Node() : m_is_eval(false) {}

        virtual void *calc(int x, int y, int z, int w)
        {
            m_is_eval = true;
            return NULL;
        }

        virtual void *calc(int idx)
        {
            m_is_eval = true;
            return NULL;
        }

        virtual void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes)
        {
            len = 0;
            buf_count = 0;
            bytes = 0;
        }

        virtual bool isLinear(const dim_t *dims) { return true; }
        virtual void reset() { m_is_eval = false;}

        virtual ~Node() {}
    };

    typedef std::shared_ptr<Node> Node_ptr;
}

}
