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
#include <af/defines.h>
#include "Node.hpp"

#include <functional>
#include <memory>
#include <sstream>
#include <string>

namespace arrayfire {
namespace cpu {

namespace jit {

template<typename T>
class BufferNode : public TNode<T> {
   protected:
    std::shared_ptr<T> m_data;
    T *m_ptr;
    unsigned m_bytes;
    dim_t m_strides[4];
    dim_t m_dims[4];
    bool m_linear_buffer;

   public:
    BufferNode()
        : TNode<T>(T(0), 0, {})
        , m_bytes(0)
        , m_strides{0, 0, 0, 0}
        , m_dims{0, 0, 0, 0}
        , m_linear_buffer(true) {}

    std::unique_ptr<common::Node> clone() final {
        return std::make_unique<BufferNode>(*this);
    }

    void setData(std::shared_ptr<T> data, unsigned bytes, dim_t data_off,
                 const dim_t *dims, const dim_t *strides,
                 const bool is_linear) {
        m_data          = data;
        m_ptr           = data.get() + data_off;
        m_bytes         = bytes;
        m_linear_buffer = is_linear;
        for (int i = 0; i < 4; i++) {
            m_strides[i] = strides[i];
            m_dims[i]    = dims[i];
        }
    }

    void setShape(af::dim4 new_shape) final {
        auto new_strides = calcStrides(new_shape);
        m_dims[0]        = new_shape[0];
        m_dims[1]        = new_shape[1];
        m_dims[2]        = new_shape[2];
        m_dims[3]        = new_shape[3];
        m_strides[0]     = new_strides[0];
        m_strides[1]     = new_strides[1];
        m_strides[2]     = new_strides[2];
        m_strides[3]     = new_strides[3];
    }

    void calc(int x, int y, int z, int w, int lim) final {
        using Tc = compute_t<T>;

        dim_t l_off = 0;
        l_off += (w < (int)m_dims[3]) * w * m_strides[3];
        l_off += (z < (int)m_dims[2]) * z * m_strides[2];
        l_off += (y < (int)m_dims[1]) * y * m_strides[1];
        T *in_ptr   = m_ptr + l_off;
        Tc *out_ptr = this->m_val.data();
        for (int i = 0; i < lim; i++) {
            out_ptr[i] =
                static_cast<Tc>(in_ptr[((x + i) < m_dims[0]) ? (x + i) : 0]);
        }
    }

    void calc(int idx, int lim) final {
        using Tc = compute_t<T>;

        T *in_ptr   = m_ptr + idx;
        Tc *out_ptr = this->m_val.data();
        for (int i = 0; i < lim; i++) {
            out_ptr[i] = static_cast<Tc>(in_ptr[i]);
        }
    }

    void getInfo(unsigned &len, unsigned &buf_count,
                 unsigned &bytes) const final {
        len++;
        buf_count++;
        bytes += m_bytes;
        return;
    }

    size_t getBytes() const final { return m_bytes; }

    void genKerName(std::string &kerString,
                    const common::Node_ids &ids) const final {
        UNUSED(kerString);
        UNUSED(ids);
    }

    void genParams(std::stringstream &kerStream, int id,
                   bool is_linear) const final {
        UNUSED(kerStream);
        UNUSED(id);
        UNUSED(is_linear);
    }

    int setArgs(int start_id, bool is_linear,
                std::function<void(int id, const void *ptr, size_t arg_size)>
                    setArg) const override {
        UNUSED(is_linear);
        UNUSED(setArg);
        return start_id++;
    }

    void genOffsets(std::stringstream &kerStream, int id,
                    bool is_linear) const final {
        UNUSED(kerStream);
        UNUSED(id);
        UNUSED(is_linear);
    }

    void genFuncs(std::stringstream &kerStream,
                  const common::Node_ids &ids) const final {
        UNUSED(kerStream);
        UNUSED(ids);
    }

    bool isLinear(const dim_t *dims) const final {
        return m_linear_buffer && dims[0] == m_dims[0] &&
               dims[1] == m_dims[1] && dims[2] == m_dims[2] &&
               dims[3] == m_dims[3];
    }

    bool isBuffer() const final { return true; }

    size_t getHash() const noexcept final {
        std::hash<const void *> ptr_hash;
        std::hash<af::dtype> aftype_hash;
        return ptr_hash(static_cast<const void *>(m_ptr)) ^
               (aftype_hash(
                    static_cast<af::dtype>(af::dtype_traits<T>::af_type))
                << 1);
    }

    /// Compares two BufferNodeBase objects for equality
    bool operator==(const BufferNode<T> &other) const noexcept {
        using std::begin;
        using std::end;
        using std::equal;
        return m_ptr == other.m_ptr && m_bytes == other.m_bytes &&
               m_linear_buffer == other.m_linear_buffer &&
               equal(begin(m_dims), end(m_dims), begin(other.m_dims)) &&
               equal(begin(m_strides), end(m_strides), begin(other.m_strides));
    };

    /// Overloads the equality operator to call comparisons between Buffer
    /// objects. Calls the BufferNodeBase equality operator if the other
    /// object is also a Buffer Node
    bool operator==(const common::Node &other) const noexcept final {
        if (other.isBuffer() && this->getType() == other.getType()) {
            return *this == static_cast<const BufferNode<T> &>(other);
        }
        return false;
    }
};

}  // namespace jit
}  // namespace cpu
}  // namespace arrayfire
