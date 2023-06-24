/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <err_oneapi.hpp>

#include <sycl/sycl.hpp>

#include <functional>
#include <memory>
#include <sstream>
#include <string>

namespace arrayfire {
namespace oneapi {

namespace {

/// Creates a string that will be used to declare the parameter of kernel
inline void generateParamDeclaration(std::stringstream& kerStream, int id,
                                     bool is_linear,
                                     const std::string& m_type_str) {
    if (is_linear) {
        kerStream << "__global " << m_type_str << " *in" << id
                  << ", dim_t iInfo" << id << "_offset, \n";
    } else {
        kerStream << "__global " << m_type_str << " *in" << id
                  << ", KParam iInfo" << id << ", \n";
    }
}

/// Calls the setArg function to set the arguments for a kernel call
template<typename T>
inline int setBufferKernelArguments(
    int start_id, bool is_linear,
    std::function<void(int id, const void* ptr, size_t arg_size,
                       bool is_buffer)>& setArg,
    const std::shared_ptr<sycl::buffer<T>>& ptr,
    const AParam<T, sycl::access_mode::read>& info) {
    setArg(start_id + 0, static_cast<const void*>(&info),
           sizeof(AParam<T, sycl::access_mode::read>), true);
    return start_id + 2;
}

/// Generates the code to calculate the offsets for a buffer
inline void generateBufferOffsets(std::stringstream& kerStream, int id,
                                  bool is_linear, const std::string& type_str) {
    UNUSED(type_str);
    std::string idx_str  = std::string("int idx") + std::to_string(id);
    std::string info_str = std::string("iInfo") + std::to_string(id);

    if (is_linear) {
        kerStream << idx_str << " = idx + " << info_str << "_offset;\n";
    } else {
        kerStream << idx_str << " = (id3 < " << info_str << ".dims[3]) * "
                  << info_str << ".strides[3] * id3 + (id2 < " << info_str
                  << ".dims[2]) * " << info_str << ".strides[2] * id2 + (id1 < "
                  << info_str << ".dims[1]) * " << info_str
                  << ".strides[1] * id1 + (id0 < " << info_str << ".dims[0]) * "
                  << info_str << ".strides[0]  * id0 + " << info_str
                  << ".offset;\n";
    }
}

/// Generates the code to read a buffer and store it in a local variable
inline void generateBufferRead(std::stringstream& kerStream, int id,
                               const std::string& type_str) {
    kerStream << type_str << " val" << id << " = in" << id << "[idx" << id
              << "];\n";
}

inline void generateShiftNodeOffsets(std::stringstream& kerStream, int id,
                                     bool is_linear,
                                     const std::string& type_str) {
    UNUSED(is_linear);
    UNUSED(type_str);
    std::string idx_str   = std::string("idx") + std::to_string(id);
    std::string info_str  = std::string("iInfo") + std::to_string(id);
    std::string id_str    = std::string("sh_id_") + std::to_string(id) + "_";
    std::string shift_str = std::string("shift") + std::to_string(id) + "_";

    for (int i = 0; i < 4; i++) {
        kerStream << "int " << id_str << i << " = __circular_mod(id" << i
                  << " + " << shift_str << i << ", " << info_str << ".dims["
                  << i << "]);\n";
    }

    kerStream << "int " << idx_str << " = (" << id_str << "3 < " << info_str
              << ".dims[3]) * " << info_str << ".strides[3] * " << id_str
              << "3;\n";
    kerStream << idx_str << " += (" << id_str << "2 < " << info_str
              << ".dims[2]) * " << info_str << ".strides[2] * " << id_str
              << "2;\n";
    kerStream << idx_str << " += (" << id_str << "1 < " << info_str
              << ".dims[1]) * " << info_str << ".strides[1] * " << id_str
              << "1;\n";
    kerStream << idx_str << " += (" << id_str << "0 < " << info_str
              << ".dims[0]) * " << id_str << "0 + " << info_str << ".offset;\n";
}

inline void generateShiftNodeRead(std::stringstream& kerStream, int id,
                                  const std::string& type_str) {
    kerStream << type_str << " val" << id << " = in" << id << "[idx" << id
              << "];\n";
}
}  // namespace
}  // namespace oneapi
}  // namespace arrayfire
