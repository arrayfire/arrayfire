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
        std::string m_op_str;
        Node_ptr m_child;
        int m_op;

    public:
        UnaryNode(const char *out_type_str, const char *name_str,
                  const char *op_str,
                  Node_ptr child, int op)
            : Node(out_type_str, name_str),
              m_op_str(op_str),
              m_child(child),
              m_op(op)
        {
        }

        bool isLinear(dim_t dims[4])
        {
            return m_child->isLinear(dims);
        }

        void genParams(std::stringstream &kerStream)
        {
            if (m_gen_param) return;
            if (!(m_child->isGenParam())) m_child->genParams(kerStream);
            m_gen_param = true;
        }

        int setArgs(cl::Kernel &ker, int id)
        {
            return m_child->setArgs(ker, id);
        }

        void genOffsets(std::stringstream &kerStream, bool is_linear)
        {
            if (m_gen_offset) return;
            if (!(m_child->isGenOffset())) m_child->genOffsets(kerStream, is_linear);
            m_gen_offset = true;
        }

        void genKerName(std::stringstream &kerStream)
        {
            m_child->genKerName(kerStream);

            // Make the dec representation of enum part of the Kernel name
            kerStream << "_" << std::setw(3) << std::setfill('0') << std::dec << m_op;
            kerStream << std::setw(3) << std::setfill('0') << std::dec << m_child->getId();
            kerStream << std::setw(3) << std::setfill('0') << std::dec << m_id << std::dec;
            m_gen_name = true;
        }

        void genFuncs(std::stringstream &kerStream)
        {
            if (m_gen_func) return;

            if (!(m_child->isGenFunc())) m_child->genFuncs(kerStream);

            kerStream << m_type_str << " val" << m_id << " = "
                      << m_op_str << "(val" << m_child->getId() << ");"
                      << "\n";

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

        void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes)
        {
            if (m_set_id) return;

            m_child->getInfo(len, buf_count, bytes);
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
            m_child->resetFlags();
        }
    };

}

}
