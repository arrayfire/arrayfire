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

    using std::shared_ptr;
    template<typename T>
    class BufferNode : public Node
    {

    protected:
        shared_ptr<T> ptr;
        dim_type off;
        dim_type strides[4];

    public:

        BufferNode(shared_ptr<T> data, dim_type *strs) :
            Node(),
            ptr(data),
            off(0)
        {
            for (int i = 0; i < 4; i++) strides[i] = strs[i];
        }

        void *calc(int x, int y, int z, int w)
        {
            off = x + y * strides[1] + z * strides[2] + w * strides[3];
            m_is_eval = true;
            return (void *)(ptr.get() + off);
        }

        void reset()
        {
            m_is_eval = false;
            off = 0;
        }
    };

}

}
