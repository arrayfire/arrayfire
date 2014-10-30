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
        std::string m_name_str;
        bool m_gen_name;
        bool m_set_arg;
        void *ptr;
        int str0, str1, str2, str3;
    public:

        BufferNode(const char *type_str,
                   const char *name_str,
                   CParam<T> &param)
            : Node(type_str),
              m_name_str(name_str),
              m_gen_name(false),
              m_set_arg(false)
        {
            ptr = (void *)param.ptr;
            str0 = (int)param.strides[0];
            str1 = (int)param.strides[1];
            str2 = (int)param.strides[2];
            str3 = (int)param.strides[3];
        }

        void genKerName(std::stringstream &kerStream, bool genInputs)
        {
            if (!genInputs) return;
            if (m_gen_name) return;

            kerStream << m_name_str;
            m_gen_name = true;
        }

        void genParams(std::stringstream &kerStream,
                       std::stringstream &annStream)
        {
            if (m_gen_param) return;
            kerStream <<  m_type_str << "* %in" << m_id << ","
                      << "i32 %str1"     << m_id << ","
                      << "i32 %str2"     << m_id << ","
                      << "i32 %str3"     << m_id << ","
                      << std::endl;
            annStream << m_type_str << "*,\n";
            annStream << "i32, i32, i32,\n";
            m_gen_param = true;
        }

        void genOffsets(std::stringstream &kerStream)
        {
            if (m_gen_offset) return;

            std::string idx_str = std::string("int idx") + toString(m_id);
            std::string info_str = std::string("iInfo") + toString(m_id);;

            kerStream << "%off3i" << m_id << " = mul i32 %id3, %str3" << m_id << "\n";
            kerStream << "%off2i" << m_id << " = mul i32 %id2, %str2" << m_id << "\n";
            kerStream << "%off1i" << m_id << " = mul i32 %id1, %str1" << m_id << "\n";

            kerStream << "%off23i" << m_id << " = add i32 %off2i"
                      << m_id << ", %off3i" << m_id << "\n";

            kerStream << "%off123i" << m_id << " = add i32 %off23i"
                      << m_id << ", %off1i" << m_id << "\n";

            kerStream << "%idxa" << m_id << " = add i32 %off123i"
                      << m_id << ", %id0" << "\n";

            kerStream << "%idx" << m_id << " = sext i32 %idxa" << m_id <<" to i64\n\n";

            m_gen_offset = true;
        }

        void genFuncs(std::stringstream &kerStream, std::stringstream &declStream)
        {
            if (m_gen_func) return;

            kerStream << "%inIdx" << m_id << " = "
                      << "getelementptr inbounds " << m_type_str << "* %in" << m_id
                      << ", i64 %idx"<< m_id << "\n";

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

        void resetFlags()
        {
            m_set_id = false;
            m_gen_func = false;
            m_gen_param = false;
            m_gen_offset = false;
            m_gen_name = false;
            m_set_arg = false;
        }

        void setArgs(std::vector<void *> &args)
        {
            if (m_set_arg) return;
            args.push_back((void *)&ptr);
            args.push_back((void *)&str1);
            args.push_back((void *)&str2);
            args.push_back((void *)&str3);
            m_set_arg = true;
        }
    };

}

}
