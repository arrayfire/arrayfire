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
#include <memory>
#include <mutex>

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

        bool isBuffer() { return true; }

        void genKerName(std::stringstream &kerStream, Node_ids ids)
        {
            kerStream << "_" << m_name_str;
            kerStream << std::setw(2) << std::setfill('0') << std::hex << ids.id << std::dec;
        }

        void genParams(std::stringstream &kerStream,
                       std::stringstream &annStream,
                       int id,
                       bool is_linear)
        {
            kerStream << m_type_str << "* %in" << id << ",\n";
            annStream << m_type_str << "*,\n";

            if (!is_linear) {
                kerStream << "i32 %dim0"     << id << ","
                          << "i32 %dim1"     << id << ","
                          << "i32 %dim2"     << id << ","
                          << "i32 %dim3"     << id << ","
                          << "\n"
                          << "i32 %str1"     << id << ","
                          << "i32 %str2"     << id << ","
                          << "i32 %str3"     << id << ","
                          << "\n";

                annStream << "i32, i32, i32, i32,\n"
                          << "i32, i32, i32,\n";
            }
        }

        void genOffsets(std::stringstream &kerStream, int id, bool is_linear)
        {
            if (!is_linear) {
                kerStream << "%b3" << id << " = icmp slt i32 %id3, %dim3" << id << "\n";
                kerStream << "%b2" << id << " = icmp slt i32 %id2, %dim2" << id << "\n";
                kerStream << "%b1" << id << " = icmp slt i32 %id1, %dim1" << id << "\n";
                kerStream << "%b0" << id << " = icmp slt i32 %id0, %dim0" << id << "\n";

                kerStream << "%c3" << id << " = zext i1 %b3" << id << " to i32\n";
                kerStream << "%c2" << id << " = zext i1 %b2" << id << " to i32\n";
                kerStream << "%c1" << id << " = zext i1 %b1" << id << " to i32\n";
                kerStream << "%c0" << id << " = zext i1 %b0" << id << " to i32\n";

                kerStream << "%d3" << id << " = mul i32 %c3" << id << ", %id3\n";
                kerStream << "%d2" << id << " = mul i32 %c2" << id << ", %id2\n";
                kerStream << "%d1" << id << " = mul i32 %c1" << id << ", %id1\n";
                kerStream << "%d0" << id << " = mul i32 %c0" << id << ", %id0\n";

                kerStream << "%off3i" << id << " = mul i32 %d3" << id
                          << ", %str3" << id << "\n";

                kerStream << "%off2i" << id << " = mul i32 %d2" << id
                          << ", %str2" << id << "\n";

                kerStream << "%off1i" << id << " = mul i32 %d1" << id
                          << ", %str1" << id << "\n";

                kerStream << "%off23i" << id << " = add i32 %off2i"
                          << id << ", %off3i" << id << "\n";

                kerStream << "%off123i" << id << " = add i32 %off23i"
                          << id << ", %off1i" << id << "\n";

                kerStream << "%idxa" << id << " = add i32 %off123i"
                          << id << ", %d0" << id << "\n";

                kerStream << "%idx" << id << " = sext i32 %idxa" << id <<" to i64\n\n";
            }
        }

        void genFuncs(std::stringstream &kerStream, str_map_t &declStrs, Node_ids ids, bool is_linear)
        {
            kerStream << "%inIdx" << ids.id << " = "
                      << "getelementptr inbounds " << m_type_str << "* %in" << ids.id
                      << ", i64 %idx";

            if (!is_linear) kerStream << ids.id;
            kerStream << "\n";

            kerStream << "%val" << ids.id << " = " << "load "
                      << m_type_str << "* %inIdx" << ids.id << "\n\n";

        }

        void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes)
        {
            len++;
            buf_count++;
            bytes += m_bytes;
            return;
        }

        void setArgs(std::vector<void *> &args, bool is_linear)
        {
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
        }
    };

}

}
