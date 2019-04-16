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

#include <iomanip>
#include <mutex>
#include <sstream>

namespace common {

template<typename DataType, typename ParamType>
class BufferNodeBase : public common::Node {
   private:
    DataType m_data;
    ParamType m_param;
    unsigned m_bytes;
    std::once_flag m_set_data_flag;
    int param_index;
    bool m_linear_buffer;

   public:
    using param_type = ParamType;
    BufferNodeBase(const char *type_str, const char *name_str)
        : Node(type_str, name_str, 0, {}) {}

    bool isBuffer() const final { return true; }

    bool requiresGlobalMemoryAccess() const final { return true; }

    void setData(ParamType param, DataType data, const unsigned bytes,
                 bool is_linear) {
        std::call_once(m_set_data_flag,
                       [this, param, data, bytes, is_linear]() {
                           m_param         = param;
                           m_data          = data;
                           m_bytes         = bytes;
                           m_linear_buffer = is_linear;
                       });
    }

    bool isLinear(dim_t dims[4]) const final {
        bool same_dims = true;
        for (int i = 0; same_dims && i < 4; i++) {
            same_dims &= (dims[i] == m_param.dims[i]);
        }
        return m_linear_buffer && same_dims;
    }

    void genKerName(std::stringstream &kerStream,
                    const common::Node_ids &ids) const final {
        kerStream << "_" << m_name_str;
        kerStream << std::setw(3) << std::setfill('0') << std::dec << ids.id
                  << std::dec;
    }

    void genParams(std::stringstream &kerStream, int id,
                   bool is_linear) const final {
        detail::generateParamDeclaration(kerStream, id, is_linear, m_type_str);
    }

    int setArgs(int start_id, bool is_linear,
                std::function<void(int id, const void *ptr, size_t arg_size)>
                    setArg) const final {
        return detail::setKernelArguments(start_id, is_linear, setArg, m_data,
                                          m_param, param_index);
    }

    void setParamIndex(int index) final { param_index = index; }
    int getParamIndex() const final { return param_index; }

    void genOffsets(std::stringstream &kerStream, int id,
                    bool is_linear) const final {
        detail::generateBufferOffsets(kerStream, id, is_linear);
    }

    void genFuncs(std::stringstream &kerStream,
                  const common::Node_ids &ids) const final {
        detail::generateBufferRead(kerStream, ids.id, m_type_str);
    }

    void getInfo(unsigned &len, unsigned &buf_count,
                 unsigned &bytes) const final {
        len++;
        buf_count++;
        bytes += m_bytes;
    }

    size_t getBytes() const final { return m_bytes; }
    ParamType &getParam() { return m_param; }
};

}  // namespace common
