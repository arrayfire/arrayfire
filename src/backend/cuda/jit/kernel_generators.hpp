/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <common/jit/Node.hpp>

#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>

namespace arrayfire {
namespace cuda {

namespace {

/// Creates a string that will be used to declare the parameter of kernel
void generateParamDeclaration(std::stringstream& kerStream, int id,
                              bool is_linear, const std::string& m_type_str) {
    if (is_linear) {
        kerStream << m_type_str << " *in" << id << "_ptr,\n";
    } else {
        kerStream << "Param<" << m_type_str << "> in" << id << ",\n";
    }
}

/// Calls the setArg function to set the arguments for a kernel call
template<typename T>
int setKernelArguments(
    int start_id, bool is_linear,
    std::function<void(int id, const void* ptr, size_t arg_size)>& setArg,
    const std::shared_ptr<T>& ptr, const Param<T>& info) {
    UNUSED(ptr);
    if (is_linear) {
        setArg(start_id, static_cast<const void*>(&info.ptr), sizeof(T*));
    } else {
        setArg(start_id, static_cast<const void*>(&info), sizeof(Param<T>));
    }
    return start_id + 1;
}

/// Generates the code to calculate the offsets for a buffer
void generateBufferOffsets(std::stringstream& kerStream, int id, bool is_linear,
                           const std::string& type_str) {
    const std::string idx_str  = std::string("idx") + std::to_string(id);
    const std::string info_str = std::string("in") + std::to_string(id);

    if (is_linear) {
        kerStream << "#define " << idx_str << " idx\n";
    } else {
        kerStream << "int " << idx_str << " = id0*(id0<" << info_str
                  << ".dims[0])*" << info_str << ".strides[0] + id1*(id1<"
                  << info_str << ".dims[1])*" << info_str
                  << ".strides[1] + id2*(id2<" << info_str << ".dims[2])*"
                  << info_str << ".strides[2] + id3*(id3<" << info_str
                  << ".dims[3])*" << info_str << ".strides[3];\n";
        kerStream << type_str << " *in" << id << "_ptr = in" << id << ".ptr;\n";
    }
}

/// Generates the code to read a buffer and store it in a local variable
void generateBufferRead(std::stringstream& kerStream, int id,
                        const std::string& type_str) {
    kerStream << type_str << " val" << id << " = in" << id << "_ptr[idx" << id
              << "];\n";
}

inline void generateShiftNodeOffsets(std::stringstream& kerStream, int id,
                                     bool is_linear,
                                     const std::string& type_str) {
    UNUSED(is_linear);
    const std::string idx_str  = std::string("idx") + std::to_string(id);
    const std::string info_str = std::string("in") + std::to_string(id);
    const std::string id_str = std::string("sh_id_") + std::to_string(id) + '_';
    const std::string shift_str =
        std::string("shift") + std::to_string(id) + '_';

    for (int i = 0; i < 4; i++) {
        kerStream << "int " << id_str << i << " = __circular_mod(id" << i
                  << " + " << shift_str << i << ", " << info_str << ".dims["
                  << i << "]);\n";
    }
    kerStream << "int " << idx_str << " = " << id_str << "0*(" << id_str << "0<"
              << info_str << ".dims[0])*" << info_str << ".strides[0] + "
              << id_str << "1*(" << id_str << "1<" << info_str << ".dims[1])*"
              << info_str << ".strides[1] + " << id_str << "2*(" << id_str
              << "2<" << info_str << ".dims[2])*" << info_str
              << ".strides[2] + " << id_str << "3*(" << id_str << "3<"
              << info_str << ".dims[3])*" << info_str << ".strides[3];\n";
    kerStream << type_str << " *in" << id << "_ptr = in" << id << ".ptr;\n";
}

inline void generateShiftNodeRead(std::stringstream& kerStream, int id,
                                  const std::string& type_str) {
    kerStream << type_str << " val" << id << " = in" << id << "_ptr[idx" << id
              << "];\n";
}

}  // namespace
}  // namespace cuda
}  // namespace arrayfire
