/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <backend.hpp>
#include <common/jit/Node.hpp>
#include <af/traits.hpp>

#include <math.hpp>
#include <types.hpp>
#include <iomanip>

namespace common {

template<typename T>
class ScalarNode : public common::Node {
   private:
    const T m_val;

   public:
    ScalarNode(T val)
        : Node(static_cast<af::dtype>(af::dtype_traits<T>::af_type), 0, {})
        , m_val(val) {}

    void genKerName(std::stringstream& kerStream,
                    const common::Node_ids& ids) const final {
        kerStream << "_" << getTypeStr();
        kerStream << std::setw(3) << std::setfill('0') << std::dec << ids.id
                  << std::dec;
    }

    void genParams(std::stringstream& kerStream, int id,
                   bool is_linear) const final {
        UNUSED(is_linear);
        kerStream << getTypeStr() << " scalar" << id << ", \n";
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
        kerStream << getTypeStr() << " val" << ids.id << " = scalar" << ids.id
                  << ";\n";
    }

    std::string getNameStr() const final { return detail::shortname<T>(false); }

    // Return the info for the params and the size of the buffers
    virtual size_t getParamBytes() const final { return sizeof(T); }
};

}  // namespace common
