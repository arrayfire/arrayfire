/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <types.hpp>
#include "Node.hpp"
#include <math.hpp>
#include <iomanip>

namespace cuda
{

namespace JIT
{
    template<typename T>
    class ScalarNode : public Node
    {
    private:
        T m_val;
    public:

        ScalarNode(T val)
            : Node(irname<T>(), afShortName<T>(false)),
              m_val(val)
        {
        }

        bool isLinear(dim_t dims[4])
        {
            return true;
        }

        void genKerName(std::stringstream &kerStream)
        {
            if (m_gen_name) return;

            kerStream << "_" << m_name_str;
            kerStream << std::setw(2) << std::setfill('0') << std::hex << m_id << std::dec;
            m_gen_name = true;
        }

        void genParams(std::stringstream &kerStream,
                       std::stringstream &annStream,
                       bool is_linear)
        {
            if (m_gen_param) return;
            kerStream << m_type_str << " %val" << m_id << ", " << std::endl;
            annStream << m_type_str << ",\n";
            m_gen_param = true;
        }

        void genOffsets(std::stringstream &kerStream, bool is_linear)
        {
            if (m_gen_offset) return;
            m_gen_offset = true;
        }

        void genFuncs(std::stringstream &kerStream, str_map_t &declStrs, bool is_linear)
        {
            if (m_gen_func) return;
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
            args.push_back((void *)&m_val);
            m_set_arg = true;
        }
    };
}

}
