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
        std::string m_op_str;
        Node *m_child;
        int m_op;

    public:
        UnaryNode(const char *out_type_str,
                  const char *op_str,
                  Node *child, int op)
            : Node(out_type_str),
              m_op_str(op_str),
              m_child(child),
              m_op(op)
        {
            m_child->addParent(this);
        }

        void replaceChild(Node *prev, Node *curr)
        {
            if (m_child == prev) m_child = curr;
            else m_child->replaceChild(prev, curr);
        }

        void genParams(std::stringstream &kerStream,
                       std::stringstream &annStream)
        {
            if (m_gen_param) return;
            if (!(m_child->isGenParam())) m_child->genParams(kerStream, annStream);
            m_gen_param = true;
        }


        void genOffsets(std::stringstream &kerStream)
        {
            if (m_gen_offset) return;
            if (!(m_child->isGenOffset())) m_child->genOffsets(kerStream);
            m_gen_offset = true;
        }

        void genKerName(std::stringstream &kerStream, bool genInputs)
        {
            if (!genInputs) {
                // Make the hex representation of enum part of the Kernel name
                kerStream << std::setw(2) << std::setfill('0') << std::hex << m_op << std::dec;
            }
            m_child->genKerName(kerStream, genInputs);
        }

        void genFuncs(std::stringstream &kerStream, std::stringstream &declStream)
        {
            if (m_gen_func) return;

            if (!(m_child->isGenFunc())) m_child->genFuncs(kerStream, declStream);

            declStream << "declare " << m_type_str << " " << m_op_str
                       << "(" << m_child->getTypeStr() << ")\n";

            kerStream << "%val" << m_id << " = call "
                      << m_type_str << " "
                      << m_op_str << "("
                      << m_child->getTypeStr() << " "
                      << "%val" << m_child->getId() << ")\n";

            m_gen_func = true;
        }

        int setId(int id)
        {
            if (m_set_id) return id;

            id = m_child->setId(id);

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
            m_child->resetFlags();
        }

        void setArgs(std::vector<void *> &args)
        {
            if (m_set_arg) return;
            m_child->setArgs(args);
            m_set_arg = true;
        }
    };

}

}
