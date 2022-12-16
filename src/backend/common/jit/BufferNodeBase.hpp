/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <backend.hpp>
#include <common/jit/Node.hpp>
#include <jit/kernel_generators.hpp>

#include <sstream>

namespace arrayfire {
namespace common {

template<typename DataType, typename ParamType>
class BufferNodeBase : public common::Node {
   private:
    DataType m_data;
    unsigned m_bytes;
    bool m_linear_buffer;

   public:
    ParamType m_param;
    BufferNodeBase(af::dtype type)
        : Node(type, 0, {}), m_bytes(0), m_linear_buffer(true) {}

    bool isBuffer() const final { return true; }

    std::unique_ptr<Node> clone() final {
        return std::make_unique<BufferNodeBase>(*this);
    }

    DataType getDataPointer() const { return m_data; }

    void setData(ParamType param, DataType data, const unsigned bytes,
                 bool is_linear) {
        m_param         = param;
        m_data          = data;
        m_bytes         = bytes;
        m_linear_buffer = is_linear;
    }

    bool isLinear(const dim_t dims[4]) const final {
        bool same_dims = true;
        for (int i = 0; same_dims && i < 4; i++) {
            same_dims &= (dims[i] == m_param.dims[i]);
        }
        return m_linear_buffer && same_dims;
    }

    void genKerName(std::string &kerString,
                    const common::Node_ids &ids) const final {
        kerString += '_';
        kerString += getNameStr();
        kerString += ',';
        kerString += std::to_string(ids.id);
    }

    void genParams(std::stringstream &kerStream, int id,
                   bool is_linear) const final {
        detail::generateParamDeclaration(kerStream, id, is_linear,
                                         getTypeStr());
    }

    int setArgs(int start_id, bool is_linear,
                std::function<void(int id, const void *ptr, size_t arg_size)>
                    setArg) const override {
        return detail::setKernelArguments(start_id, is_linear, setArg, m_data,
                                          m_param);
    }

    void genOffsets(std::stringstream &kerStream, int id,
                    bool is_linear) const final {
        detail::generateBufferOffsets(kerStream, id, is_linear, getTypeStr());
    }

    void genFuncs(std::stringstream &kerStream,
                  const common::Node_ids &ids) const final {
        detail::generateBufferRead(kerStream, ids.id, getTypeStr());
    }

    void getInfo(unsigned &len, unsigned &buf_count,
                 unsigned &bytes) const final {
        len++;
        buf_count++;
        bytes += m_bytes;
    }

    size_t getBytes() const final { return m_bytes; }

    size_t getHash() const noexcept override {
        size_t out = 0;
        auto ptr   = m_data.get();
        memcpy(&out, &ptr, std::max(sizeof(Node *), sizeof(size_t)));
        return out;
    }

    /// Compares two BufferNodeBase objects for equality
    bool operator==(
        const BufferNodeBase<DataType, ParamType> &other) const noexcept;

    /// Overloads the equality operator to call comparisons between Buffer
    /// objects. Calls the BufferNodeBase equality operator if the other
    /// object is also a Buffer Node
    bool operator==(const common::Node &other) const noexcept final {
        if (other.isBuffer()) {
            return *this ==
                   static_cast<const BufferNodeBase<DataType, ParamType> &>(
                       other);
        }
        return false;
    }
};

}  // namespace common
}  // namespace arrayfire
