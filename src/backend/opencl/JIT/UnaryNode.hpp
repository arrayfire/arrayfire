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

    class UnaryNode : public Node
    {
    private:
        const std::string m_op_str;
        const int m_op;

    public:
        UnaryNode(const char *out_type_str, const char *name_str,
                  const char *op_str,
                  Node_ptr child, int op)
            : Node(out_type_str, name_str, child->getHeight() + 1, {child}),
              m_op_str(op_str),
              m_op(op)
        {
        }

        void genKerName(std::stringstream &kerStream, Node_ids ids)
        {
            // Make the dec representation of enum part of the Kernel name
            kerStream << "_" << std::setw(3) << std::setfill('0') << std::dec << m_op;
            kerStream << std::setw(3) << std::setfill('0') << std::dec << ids.child_ids[0];
            kerStream << std::setw(3) << std::setfill('0') << std::dec << ids.id << std::dec;
        }

        void genFuncs(std::stringstream &kerStream, Node_ids ids)
        {
            kerStream << m_type_str << " val" << ids.id << " = "
                      << m_op_str << "(val" << ids.child_ids[0] << ");"
                      << "\n";
        }
    };

}

}
