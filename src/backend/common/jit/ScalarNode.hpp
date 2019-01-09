/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <common/jit/Node.hpp>

#include <math.hpp>
#include <types.hpp>
#include <iomanip>

namespace common {

template <typename T>
class ScalarNode : public common::Node {
   private:
    const T m_val;

   public:
    ScalarNode(T val)
        : Node(detail::getFullName<T>(), detail::shortname<T>(false), 0, {})
        , m_val(val) {}

    void genKerName(std::stringstream& kerStream,
                    const common::Node_ids& ids) const final {
        kerStream << "_" << m_name_str;
        kerStream << std::setw(3) << std::setfill('0') << std::dec << ids.id
                  << std::dec;
    }

    void genParams(std::stringstream& kerStream, int id,
                   bool is_linear) const final {
        UNUSED(is_linear);
        kerStream << m_type_str << " scalar" << id << ", \n";
    }

    int setArgs(int start_id, bool is_linear,
                std::function<void(int id, const void* ptr, size_t arg_size)>
                    setArg) const final {
        UNUSED(is_linear);
        setArg(start_id, static_cast<const void*>(&m_val), sizeof(T));
        return start_id + 1;
    }

    void genFuncs(std::stringstream& kerStream,
                  const common::Node_ids& ids) const final {
        kerStream << m_type_str << " val" << ids.id << " = scalar" << ids.id
                  << ";\n";
    }

    // Return the info for the params and the size of the buffers
    virtual short getParamBytes() const final {
        return static_cast<short>(sizeof(T));
    }
};

}  // namespace common
