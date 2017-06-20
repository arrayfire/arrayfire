/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include "Node.hpp"
#include <iomanip>
#include <mutex>

namespace cuda
{

namespace JIT
{


    template<typename T>
    class BufferNode : public Node
    {
    private:
        std::shared_ptr<T> m_data;
        Param<T> m_param;
        unsigned m_bytes;
        std::once_flag m_set_data_flag;
        bool m_linear_buffer;

    public:

        BufferNode(const char *type_str,
                   const char *name_str)
            : Node(type_str, name_str, 0, {})
        {
        }

        bool isBuffer() { return true; }

        void setData(Param<T> param, std::shared_ptr<T> data, const unsigned bytes, bool is_linear)
        {
            std::call_once(m_set_data_flag, [this, param, data, bytes, is_linear]() {
                    m_param = param;
                    m_data = data;
                    m_bytes = bytes;
                    m_linear_buffer = is_linear;
                });
        }

        bool isLinear(dim_t dims[4])
        {
            bool same_dims = true;
            for (int i = 0; same_dims && i < 4; i++) {
                same_dims &= (dims[i] == m_param.dims[i]);
            }
            return m_linear_buffer && same_dims;
        }

        void genKerName(std::stringstream &kerStream, Node_ids ids)
        {
            kerStream << "_" << m_name_str;
            kerStream << std::setw(3) << std::setfill('0') << std::dec << ids.id << std::dec;
        }

        void genParams(std::stringstream &kerStream, int id, bool is_linear)
        {
            if (is_linear) {
                kerStream << m_type_str << " *in" << id << "_ptr,\n";
            } else {
                kerStream << "Param<" << m_type_str << "> in" << id
                          << ",\n";
            }
        }

        void setArgs(std::vector<void *> &args, bool is_linear)
        {
            if (is_linear) {
                args.push_back((void *)&m_param.ptr);
            } else {
                args.push_back((void *)&m_param);
            }
        }

        void genOffsets(std::stringstream &kerStream, int id, bool is_linear)
        {
            std::string idx_str = std::string("int idx") + std::to_string(id);

            if (is_linear) {
                kerStream << idx_str << " = idx;\n";
            } else {
                std::string info_str = std::string("in") + std::to_string(id);
                kerStream << idx_str << " = "
                          << "(id3 < " << info_str << ".dims[3]) * "
                          << info_str << ".strides[3] * id3 + "
                          << "(id2 < " << info_str << ".dims[2]) * "
                          << info_str << ".strides[2] * id2 + "
                          << "(id1 < " << info_str << ".dims[1]) * "
                          << info_str << ".strides[1] * id1 + "
                          << "(id0 < " << info_str << ".dims[0]) * "
                          << "id0;"
                          << "\n";
                kerStream << m_type_str << " *in" << id << "_ptr = in" << id << ".ptr;\n";
            }
        }

        void genFuncs(std::stringstream &kerStream, Node_ids ids)
        {
            kerStream << m_type_str << " val" << ids.id << " = "
                      << "in" << ids.id << "_ptr[idx" << ids.id << "];"
                      << "\n";
        }

        void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes)
        {
            len++;
            buf_count++;
            bytes += m_bytes;
            return;
        }
    };

}

}
