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

namespace cuda
{

namespace JIT
{

    template <typename T>
    class ScalarNode : public Node
    {
    private:
        const T m_val;

    public:

        ScalarNode(T val)
            : Node(getFullName<T>(), shortname<T>(false), 0, {}),
              m_val(val)
        {
        }

        void genKerName(std::stringstream &kerStream, Node_ids ids)
        {
            kerStream << "_" << m_name_str;
            kerStream << std::setw(3) << std::setfill('0') << std::dec << ids.id << std::dec;
        }

        void genParams(std::stringstream &kerStream, int id, bool is_linear)
        {
            kerStream << m_type_str << " scalar" << id << ", " << "\n";
        }

        void setArgs(std::vector<void *> &args, bool is_linear)
        {
            args.push_back((void *)&m_val);
        }

        void genFuncs(std::stringstream &kerStream, Node_ids ids)
        {
            kerStream << m_type_str << " val" << ids.id << " = "
                      << "scalar" << ids.id << ";"
                      << "\n";
        }
    };

}

}
