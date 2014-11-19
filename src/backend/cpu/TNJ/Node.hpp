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

        virtual void reset() { m_is_eval = false;}

        virtual ~Node() {}
    };

    typedef std::shared_ptr<Node> Node_ptr;
}

}
