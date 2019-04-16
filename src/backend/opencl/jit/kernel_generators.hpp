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

#include <functional>
#include <memory>
#include <sstream>
#include <string>

namespace opencl {

namespace {

/// Creates a string that will be used to declare the parameter of kernel
inline void generateParamDeclaration(std::stringstream& kerStream, int id,
                                     bool is_linear,
                                     const std::string& m_type_str) {
    kerStream << "__global " << m_type_str << " *in" << id << "_ptr, dim_t in"
              << id << "_offset, ";
    if (is_linear) {
        kerStream << "\n";
    } else {
        kerStream << "int in_index" << id << ", \n";
    }
}

/// Calls the setArg function to set the arguments for a kernel call
inline int setKernelArguments(
    int start_id, bool is_linear,
    std::function<void(int id, const void* ptr, size_t arg_size)>& setArg,
    const std::shared_ptr<cl::Buffer>& ptr, const KParam& info,
    const int& param_index) {
    cl_mem& buf = (*ptr)();
    setArg(start_id++, static_cast<const void*>(&buf), sizeof(cl_mem));
    setArg(start_id++, static_cast<const void*>(&info.offset), sizeof(dim_t));
    if (is_linear == false) {
        setArg(start_id++, static_cast<const void*>(&param_index), sizeof(int));
    }
    return start_id;
}

/// Generates the code to calculate the offsets for a buffer
inline void generateBufferOffsets(std::stringstream& kerStream, int id,
                                  bool is_linear) {
    std::string idx_str    = std::string("dim_t idx") + std::to_string(id);
    std::string offset_str = std::string("in") + std::to_string(id) + "_offset";

    if (is_linear) {
        kerStream << idx_str << " = idx + " << offset_str << ";\n";
    } else {
        std::string in_index     = "in_index" + std::to_string(id);
        std::string block_offset = "block_offsets[" + in_index + "]";
        std::string in_param     = "params[" + in_index + "]";

        kerStream << idx_str << " = " << block_offset << "\n + ((id1 < "
                  << in_param << ".dims[1]) * " << in_param
                  << ".strides[1] * id1)"
                  << "\n + ((id0 < " << in_param << ".dims[0]) * id0) + "
                  << offset_str << ";\n";
    }
}

/// Generates the code to read a buffer and store it in a local variable
inline void generateBufferRead(std::stringstream& kerStream, int id,
                               const std::string& type_str) {
    kerStream << type_str << " val" << id << " = in" << id << "_ptr[idx" << id
              << "];\n";
}

inline void generateShiftNodeOffsets(std::stringstream& kerStream, int id) {
    std::string idx_str   = std::string("idx") + std::to_string(id);
    std::string info_str  = std::string("in") + std::to_string(id);
    std::string id_str    = std::string("sh_id_") + std::to_string(id) + "_";
    std::string shift_str = std::string("shift") + std::to_string(id) + "_";

    kerStream << "KParam " << info_str << " = params[in_index" << id << "];\n";
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
              << ".dims[0]) * " << id_str << "0 + " << info_str << "_offset;\n";
}

inline void generateShiftNodeRead(std::stringstream& kerStream, int id,
                                  const std::string& type_str) {
    kerStream << type_str << " val" << id << " = in" << id << "_ptr[idx" << id
              << "];\n";
}
}  // namespace
}  // namespace opencl
