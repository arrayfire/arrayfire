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
        unsigned m_bytes;
        dim_type off;
        dim_type strides[4];
        dim_type dims[4];

    public:

        BufferNode(shared_ptr<T> data,
                   unsigned bytes,
                   dim_type data_off,
                   const dim_type *dms,
                   const dim_type *strs) :
            Node(),
            ptr(data),
            m_bytes(bytes),
            off(data_off)
        {
            for (int i = 0; i < 4; i++) {
                strides[i] = strs[i];
                dims[i] = dms[i];
            }
        }

        void *calc(int x, int y, int z, int w)
        {
            dim_type l_off = 0;
            l_off += (w < dims[3]) * w * strides[3];
            l_off += (z < dims[2]) * z * strides[2];
            l_off += (y < dims[1]) * y * strides[1];
            l_off += (x < dims[0]) * x;
            return (void *)(ptr.get() + off + l_off);
        }

        void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes)
        {
            if (m_is_eval) return;

            len++;
            buf_count++;
            bytes += m_bytes;
            m_is_eval = true;
            return;
        }

        void reset()
        {
            m_is_eval = false;
            off = 0;
        }
    };

}

}
