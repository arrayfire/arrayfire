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

    class UnaryNode : public Node
    {
    private:
        const std::string m_op_str;
        const int m_op;
        const bool m_is_check;

    public:
        UnaryNode(const char *out_type_str, const char *name_str,
                  const std::string &op_str,
                  Node_ptr child, int op, bool is_check=false)
            : Node(out_type_str, name_str, child->getHeight() + 1, {child}),
              m_op_str(op_str),
              m_op(op),
              m_is_check(is_check)
        {
        }

        void genKerName(std::stringstream &kerStream, Node_ids ids)
        {
            // Make the hex representation of enum part of the Kernel name
            kerStream << "_" << std::setw(2) << std::setfill('0') << std::hex << m_op;
            kerStream << std::setw(2) << std::setfill('0') << std::hex << ids.child_ids[0];
            kerStream << std::setw(2) << std::setfill('0') << std::hex << ids.id << std::dec;
        }

        void genFuncs(std::stringstream &kerStream, str_map_t &declStrs, Node_ids ids, bool is_linear)
        {
            std::stringstream declStream;

            if (m_is_check) {
                declStream << "declare " << "i32 " << m_op_str
                           << "(" << m_children[0]->getTypeStr() << ")\n";
            } else {
                declStream << "declare " << m_type_str << " " << m_op_str
                           << "(" << m_children[0]->getTypeStr() << ")\n";
            }

            declStrs[declStream.str()] = true;

            if (m_is_check) {
                kerStream << "%tmp" << ids.id << " = call i32 "
                          << m_op_str << "("
                          << m_children[0]->getTypeStr() << " "
                          << "%val" << ids.child_ids[0] << ")\n";

                if (m_type_str[0] == 'i') {
                    kerStream << "%val" << ids.id << " = "
                              << "trunc i32 %tmp" << ids.id << " to " << m_type_str << "\n";
                } else {
                    kerStream << "%val" << ids.id << " = "
                              << "sitofp i32 %tmp" << ids.id << " to " << m_type_str << "\n";
                }

            } else {
                kerStream << "%val" << ids.id << " = call "
                          << m_type_str << " "
                          << m_op_str << "("
                          << m_children[0]->getTypeStr() << " "
                          << "%val" << ids.child_ids[0] << ")\n";
            }
        }
    };

}

}
