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
        bool m_is_linear;
        dim_t m_off;
        dim_t m_strides[4];
        dim_t m_dims[4];
    public:

        BufferNode(shared_ptr<T> data,
                   unsigned bytes,
                   dim_t data_off,
                   const dim_t *dms,
                   const dim_t *strs,
                   const bool is_linear) :
            Node(),
            ptr(data),
            m_bytes(bytes),
            m_is_linear(is_linear),
            m_off(data_off)
        {
            for (int i = 0; i < 4; i++) {
                m_strides[i] = strs[i];
                m_dims[i] = dms[i];
            }
        }

        void *calc(int x, int y, int z, int w)
        {
            dim_t l_off = 0;
            l_off += (w < (int)m_dims[3]) * w * m_strides[3];
            l_off += (z < (int)m_dims[2]) * z * m_strides[2];
            l_off += (y < (int)m_dims[1]) * y * m_strides[1];
            l_off += (x < (int)m_dims[0]) * x;
            return (void *)(ptr.get() + m_off + l_off);
        }

        void *calc(int idx)
        {
            return (void *)(ptr.get() + idx + m_off);
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
        }

        bool isLinear(const dim_t *dims)
        {
            return m_is_linear &&
                dims[0] == m_dims[0] &&
                dims[1] == m_dims[1] &&
                dims[2] == m_dims[2] &&
                dims[3] == m_dims[3];
        }
    };

}

}
