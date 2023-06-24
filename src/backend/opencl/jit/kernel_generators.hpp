/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <sstream>
#include <string>

namespace arrayfire {
namespace opencl {

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
inline int setBufferKernelArguments(
    int start_id, bool is_linear,
    std::function<void(int id, const void* ptr, size_t arg_size,
                       bool is_buffer)>& setArg,
    const std::shared_ptr<cl::Buffer>& ptr, const KParam& info) {
    setArg(start_id + 0, static_cast<const void*>(&ptr.get()->operator()()),
           sizeof(cl_mem), true);
    if (is_linear) {
        setArg(start_id + 1, static_cast<const void*>(&info.offset),
               sizeof(dim_t), true);
    } else {
        setArg(start_id + 1, static_cast<const void*>(&info), sizeof(KParam),
               true);
    }
    return start_id + 2;
}

/// Generates the code to calculate the offsets for a buffer
inline void generateBufferOffsets(std::stringstream& kerStream, int id,
                                  bool is_linear, const std::string& type_str) {
    UNUSED(type_str);
    const std::string idx_str  = std::string("idx") + std::to_string(id);
    const std::string info_str = std::string("iInfo") + std::to_string(id);
    const std::string in_str   = std::string("in") + std::to_string(id);

    if (is_linear) {
        kerStream << in_str << " += " << info_str << "_offset;\n"
                  << "#define " << idx_str << " idx\n";
    } else {
        kerStream << "int " << idx_str << " = id0*(id0<" << info_str
                  << ".dims[0])*" << info_str << ".strides[0] + id1*(id1<"
                  << info_str << ".dims[1])*" << info_str
                  << ".strides[1] + id2*(id2<" << info_str << ".dims[2])*"
                  << info_str << ".strides[2] + id3*(id3<" << info_str
                  << ".dims[3])*" << info_str << ".strides[3] + " << info_str
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
    const std::string idx_str  = std::string("idx") + std::to_string(id);
    const std::string info_str = std::string("iInfo") + std::to_string(id);
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
              << info_str << ".dims[3])*" << info_str << ".strides[3] + "
              << info_str << ".offset;\n";
}

inline void generateShiftNodeRead(std::stringstream& kerStream, int id,
                                  const std::string& type_str) {
    kerStream << type_str << " val" << id << " = in" << id << "[idx" << id
              << "];\n";
}
}  // namespace
}  // namespace opencl
}  // namespace arrayfire
