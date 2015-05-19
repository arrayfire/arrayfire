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
#include <math.hpp>
#include <types.hpp>
#include <iomanip>

namespace opencl
{

namespace JIT
{

    template <typename T>
    class ScalarNode : public Node
    {
    private:
        const T m_val;
        bool m_set_arg;

    public:

        ScalarNode(T val)
            : Node(dtype_traits<T>::getName(), shortname<T>(false)),
              m_val(val),
              m_set_arg(false)
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
            kerStream << std::setw(3) << std::setfill('0') << std::dec << m_id << std::dec;
            m_gen_name = true;
        }

        void genParams(std::stringstream &kerStream)
        {
            if (m_gen_param) return;
            kerStream << m_type_str << " scalar" << m_id << ", " << "\n";
            m_gen_param = true;
        }

        int setArgs(cl::Kernel &ker, int id)
        {
            if (m_set_arg) return id;
            ker.setArg(id, m_val);
            m_set_arg = true;
            return id + 1;
        }

        void genOffsets(std::stringstream &kerStream, bool is_linear)
        {
            if (m_gen_offset) return;
            m_gen_offset = true;
        }

        void genFuncs(std::stringstream &kerStream)
        {
            if (m_gen_func) return;

            kerStream << m_type_str << " val" << m_id << " = "
                      << "scalar" << m_id << ";"
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
    };

}

}
