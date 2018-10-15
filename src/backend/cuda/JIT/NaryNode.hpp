/*******************************************************
 * Copyright (c) 2018, ArrayFire
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

    class NaryNode : public Node
    {
    private:
        const int m_num_children;
        const int m_op;
        const std::string m_op_str;

    public:
        NaryNode(const char *out_type_str,
                 const char *name_str,
                 const char *op_str,
                 const int num_children,
                 const std::array<Node_ptr, MAX_CHILDREN> &children,
                 const int op, const int height)
            : Node(out_type_str, name_str, height, children),
              m_num_children(num_children),
              m_op(op),
              m_op_str(op_str)
        {
        }

        void genKerName(std::stringstream &kerStream, Node_ids ids) const final
        {
            // Make the dec representation of enum part of the Kernel name
            kerStream << "_" << std::setw(3) << std::setfill('0') << std::dec << m_op;
            for (int i = 0; i < m_num_children; i++) {
                kerStream << std::setw(3)
                          << std::setfill('0')
                          << std::dec
                          << ids.child_ids[i];
            }
            kerStream << std::setw(3) << std::setfill('0') << std::dec << ids.id << std::dec;
        }

        void genFuncs(std::stringstream &kerStream, Node_ids ids) const final
        {
            kerStream << m_type_str << " val" << ids.id << " = " << m_op_str << "(";
            for (int i = 0; i < m_num_children; i++) {
                if (i > 0) kerStream << ", ";
                kerStream << "val" << ids.child_ids[i];
            }
            kerStream << ");\n";
        }
    };

}

}
