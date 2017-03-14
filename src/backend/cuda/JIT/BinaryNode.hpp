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

    class BinaryNode : public Node
    {
    private:
        const std::string m_op_str;
        const int m_op;
        const int m_call_type;

    public:
        BinaryNode(const char *out_type_str, const char *name_str,
                   const std::string &op_str,
                   Node_ptr lhs, Node_ptr rhs, int op, int call_type)
            : Node(out_type_str, name_str, std::max(lhs->getHeight(), rhs->getHeight()) + 1, {lhs, rhs}),
              m_op_str(op_str),
              m_op(op),
              m_call_type(call_type)
        {
        }

        void genKerName(std::stringstream &kerStream, Node_ids ids)
        {
            // Make the hex representation of enum part of the Kernel name
            kerStream << "_" << std::setw(2) << std::setfill('0') << std::hex << m_op;
            kerStream << std::setw(2) << std::setfill('0') << std::hex << ids.child_ids[0];
            kerStream << std::setw(2) << std::setfill('0') << std::hex << ids.child_ids[1];
            kerStream << std::setw(2) << std::setfill('0') << std::hex << ids.id << std::dec;
        }

        void genFuncs(std::stringstream &kerStream, str_map_t &declStrs, Node_ids ids, bool is_linear)
        {
            if (m_call_type == 0) {
                std::stringstream declStream;
                declStream << "declare " << m_type_str << " " << m_op_str
                           << "(" << m_children[0]->getTypeStr() << " , "
                           << m_children[1]->getTypeStr() << ")\n";
                declStrs[declStream.str()] = true;

                kerStream << "%val" << ids.id << " = call "
                          << m_type_str << " "
                          << m_op_str << "("
                          << m_children[0]->getTypeStr() << " "
                          << "%val" << ids.child_ids[0] << ", "
                          << m_children[1]->getTypeStr() << " "
                          << "%val" << ids.child_ids[1] << ")\n";

            } else {
                if (m_call_type == 1) {
                    // arithmetic operations
                    kerStream << "%val" << ids.id << " = "
                              << m_op_str << " "
                              << m_type_str << " "
                              << "%val" << ids.child_ids[0] << ", "
                              << "%val" << ids.child_ids[1] << "\n";
                } else {
                    // logical operators
                    kerStream << "%tmp" << ids.id << " = "
                              << m_op_str << " "
                              << m_children[0]->getTypeStr() << " "
                              << "%val" << ids.child_ids[0] << ", "
                              << "%val" << ids.child_ids[1] << "\n";

                    kerStream << "%val" << ids.id << " = "
                              << "zext i1 %tmp" << ids.id << " to i8\n";

                }
            }
        }
    };

}

}
