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

namespace opencl
{

namespace JIT
{


    class BufferNode : public Node
    {
    private:
        std::shared_ptr<cl::Buffer> m_data;
        KParam m_info;
        unsigned m_bytes;
        bool m_linear_buffer;

    public:

        BufferNode(const char *type_str,
                   const char *name_str)
            : Node(type_str, name_str)
        {
        }

        bool isBuffer() { return true; }

        ~BufferNode()
        {
        }

        void setData(KParam info, std::shared_ptr<cl::Buffer> data, const unsigned bytes, bool is_linear)
        {
            m_info = info;
            m_data = data;
            m_bytes = bytes;
            m_linear_buffer = is_linear;
        }

        bool isLinear(dim_t dims[4])
        {
            if (!m_set_is_linear) {
                bool same_dims = true;
                for (int i = 0; same_dims && i < 4; i++) {
                    same_dims &= (dims[i] == m_info.dims[i]);
                }
                m_set_is_linear = true;
                m_linear = m_linear_buffer && same_dims;
            }
            return m_linear;
        }

        void genKerName(std::stringstream &kerStream)
        {
            if (m_gen_name) return;

            kerStream << "_" << m_name_str;
            kerStream << std::setw(3) << std::setfill('0') << std::dec << m_id << std::dec;
            m_gen_name = true;
        }

        void genParams(std::stringstream &kerStream)
        {
            if (m_gen_param) return;
            kerStream << "__global " << m_type_str << " *in" << m_id
                      << ", KParam iInfo" << m_id << ", " << "\n";
            m_gen_param = true;
        }

        int setArgs(cl::Kernel &ker, int id)
        {
            if (m_set_arg) return id;

            ker.setArg(id + 0, *m_data);
            ker.setArg(id + 1,  m_info);

            m_set_arg = true;
            return id + 2;
        }

        void genOffsets(std::stringstream &kerStream, bool is_linear)
        {
            if (m_gen_offset) return;

            std::string idx_str = std::string("int idx") + std::to_string(m_id);
            std::string info_str = std::string("iInfo") + std::to_string(m_id);;

            if (!is_linear) {
                kerStream << idx_str << " = "
                          << "(id3 < " << info_str << ".dims[3]) * "
                          << info_str << ".strides[3] * id3 + "
                          << "(id2 < " << info_str << ".dims[2]) * "
                          << info_str << ".strides[2] * id2 + "
                          << "(id1 < " << info_str << ".dims[1]) * "
                          << info_str << ".strides[1] * id1 + "
                          << "(id0 < " << info_str << ".dims[0]) * "
                          << "id0 + " << info_str << ".offset;"
                          << "\n";
            } else {
                kerStream << idx_str << " = idx + " << info_str << ".offset;" << "\n";
            }

            m_gen_offset = true;
        }

        void genFuncs(std::stringstream &kerStream)
        {
            if (m_gen_func) return;

            kerStream << m_type_str << " val" << m_id << " = "
                      << "in" << m_id << "[idx" << m_id << "];"
                      << "\n";

            m_gen_func = true;
        }

        int setId(int id)
        {
            if (m_set_id) return id;

            m_id = id;
            m_set_id = true;

            return m_id + 1;
        }

        void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes)
        {
            if (m_set_id) return;

            len++;
            buf_count++;
            bytes += m_bytes;
            m_set_id = true;
            return;
        }


        void resetFlags()
        {
            if (m_set_id) resetCommonFlags();
        }
    };

}

}
