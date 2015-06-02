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
        dim_t off;
        dim_t strides[4];
        dim_t dims[4];

    public:

        BufferNode(shared_ptr<T> data,
                   unsigned bytes,
                   dim_t data_off,
                   const dim_t *dms,
                   const dim_t *strs) :
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
            dim_t l_off = 0;
            l_off += (w < (int)dims[3]) * w * strides[3];
            l_off += (z < (int)dims[2]) * z * strides[2];
            l_off += (y < (int)dims[1]) * y * strides[1];
            l_off += (x < (int)dims[0]) * x;
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

        void reset(bool reset_off=true)
        {
            m_is_eval = false;
            if (reset_off) off = 0;
        }
    };

}

}
