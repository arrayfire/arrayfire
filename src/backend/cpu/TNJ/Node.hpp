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
#include <memory>

namespace cpu
{

namespace TNJ
{

    class Node
    {

    protected:

        int x, y, z, w;
        bool m_is_eval;
        bool m_linear;
        bool m_set_is_linear;


        void resetCommonFlags()
        {
            x = -1;
            y = -1;
            z = -1;
            w = -1;
            m_is_eval = false;
            m_linear = false;
            m_set_is_linear = false;
        }

        bool calcCurrent(int xc)
        {
            bool res = (x == xc);
            x = xc;
            return !res;
        }

        bool calcCurrent(int xc, int yc, int zc, int wc)
        {
            bool res = (xc == x) && (yc == y) && (zc == z) && (wc == w);
            x = xc;
            y = yc;
            z = zc;
            w = wc;
            return !res;
        }

    public:
        Node() :
            x(-1),
            y(-1),
            z(-1),
            w(-1),
            m_is_eval(false),
            m_linear(false),
            m_set_is_linear(false)
        {}

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
        virtual void reset() { resetCommonFlags(); }

        virtual ~Node() {}
    };

    typedef std::shared_ptr<Node> Node_ptr;
}

}
