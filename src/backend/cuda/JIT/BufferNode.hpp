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

namespace cuda
{

namespace JIT
{

    template <typename T>
    static inline std::string toString(T val)
    {
        std::stringstream s;
        s << val;
        return s.str();
    }

    template<typename T>
    class BufferNode : public Node
    {
    private:
        // Keep the shared pointer for reference counting
        shared_ptr<T> sptr;
        CParam<T> m_param;
        unsigned m_bytes;

        bool m_linear;
    public:

        BufferNode(const char *type_str,
                   const char *name_str,
                   shared_ptr<T> data,
                   CParam<T> param,
                   unsigned bytes,
                   bool is_linear)
            : Node(type_str, name_str),
              sptr(data),
              m_param(param),
              m_bytes(bytes),
              m_linear(is_linear)
        {
        }

        bool isLinear(dim_t dims[4])
        {
            bool same_dims = true;
            for (int i = 0; same_dims && i < 4; i++) {
                same_dims &= (dims[i] == m_param.dims[i]);
            }
            return m_linear && same_dims;
        }

        void genKerName(std::stringstream &kerStream)
        {
            if (m_gen_name) return;

            kerStream << "_" << m_name_str;
            kerStream << std::setw(2) << std::setfill('0') << std::hex << m_id << std::dec;
            m_gen_name = true;
        }

        void genParams(std::stringstream &kerStream,
                       std::stringstream &annStream, bool is_linear)
        {
            if (m_gen_param) return;
            kerStream << m_type_str << "* %in" << m_id << ",\n";
            annStream << m_type_str << "*,\n";

            if (!is_linear) {
                kerStream << "i32 %dim0"     << m_id << ","
                          << "i32 %dim1"     << m_id << ","
                          << "i32 %dim2"     << m_id << ","
                          << "i32 %dim3"     << m_id << ","
                          << "\n"
                          << "i32 %str1"     << m_id << ","
                          << "i32 %str2"     << m_id << ","
                          << "i32 %str3"     << m_id << ","
                          << "\n";

                annStream << "i32, i32, i32, i32,\n"
                          << "i32, i32, i32,\n";
            }

            m_gen_param = true;
        }

        void genOffsets(std::stringstream &kerStream, bool is_linear)
        {
            if (m_gen_offset) return;

            if (!is_linear) {
                kerStream << "%b3" << m_id << " = icmp slt i32 %id3, %dim3" << m_id << "\n";
                kerStream << "%b2" << m_id << " = icmp slt i32 %id2, %dim2" << m_id << "\n";
                kerStream << "%b1" << m_id << " = icmp slt i32 %id1, %dim1" << m_id << "\n";
                kerStream << "%b0" << m_id << " = icmp slt i32 %id0, %dim0" << m_id << "\n";

                kerStream << "%c3" << m_id << " = zext i1 %b3" << m_id << " to i32\n";
                kerStream << "%c2" << m_id << " = zext i1 %b2" << m_id << " to i32\n";
                kerStream << "%c1" << m_id << " = zext i1 %b1" << m_id << " to i32\n";
                kerStream << "%c0" << m_id << " = zext i1 %b0" << m_id << " to i32\n";

                kerStream << "%d3" << m_id << " = mul i32 %c3" << m_id << ", %id3\n";
                kerStream << "%d2" << m_id << " = mul i32 %c2" << m_id << ", %id2\n";
                kerStream << "%d1" << m_id << " = mul i32 %c1" << m_id << ", %id1\n";
                kerStream << "%d0" << m_id << " = mul i32 %c0" << m_id << ", %id0\n";

                kerStream << "%off3i" << m_id << " = mul i32 %d3" << m_id
                          << ", %str3" << m_id << "\n";

                kerStream << "%off2i" << m_id << " = mul i32 %d2" << m_id
                          << ", %str2" << m_id << "\n";

                kerStream << "%off1i" << m_id << " = mul i32 %d1" << m_id
                          << ", %str1" << m_id << "\n";

                kerStream << "%off23i" << m_id << " = add i32 %off2i"
                          << m_id << ", %off3i" << m_id << "\n";

                kerStream << "%off123i" << m_id << " = add i32 %off23i"
                          << m_id << ", %off1i" << m_id << "\n";

                kerStream << "%idxa" << m_id << " = add i32 %off123i"
                          << m_id << ", %d0" << m_id << "\n";

                kerStream << "%idx" << m_id << " = sext i32 %idxa" << m_id <<" to i64\n\n";
            }

            m_gen_offset = true;
        }

        void genFuncs(std::stringstream &kerStream, str_map_t &declStrs, bool is_linear)
        {
            if (m_gen_func) return;

            kerStream << "%inIdx" << m_id << " = "
                      << "getelementptr inbounds " << m_type_str << "* %in" << m_id
                      << ", i64 %idx";

            if (!is_linear) kerStream << m_id;
            kerStream << "\n";

            kerStream << "%val" << m_id << " = " << "load "
                      << m_type_str << "* %inIdx" << m_id << "\n\n";

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
            m_set_id = false;
            m_gen_func = false;
            m_gen_param = false;
            m_gen_offset = false;
            m_gen_name = false;
            m_set_arg = false;
        }

        void setArgs(std::vector<void *> &args, bool is_linear)
        {
            if (m_set_arg) return;
            args.push_back((void *)&(m_param.ptr));

            if (!is_linear) {
                args.push_back((void *)&m_param.dims[0]);
                args.push_back((void *)&m_param.dims[1]);
                args.push_back((void *)&m_param.dims[2]);
                args.push_back((void *)&m_param.dims[3]);
                args.push_back((void *)&m_param.strides[1]);
                args.push_back((void *)&m_param.strides[2]);
                args.push_back((void *)&m_param.strides[3]);
            }
            m_set_arg = true;
        }
    };

}

}
