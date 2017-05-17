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
            : Node(irname<T>(), afShortName<T>(false), 0, {}),
              m_val(val)
        {
        }

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
            kerStream << m_type_str << " %val" << id << ", " << std::endl;
            annStream << m_type_str << ",\n";
        }

        void setArgs(std::vector<void *> &args, bool is_linear)
        {
            args.push_back((void *)&m_val);
        }
    };
}

}
