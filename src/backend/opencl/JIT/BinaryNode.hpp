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

    class BinaryNode : public Node
    {
    private:
        std::string m_op_str;
        Node_ptr m_lhs, m_rhs;
        int m_op;

    public:
        BinaryNode(const char *out_type_str, const char *name_str,
                   const char *op_str,
                   Node_ptr lhs, Node_ptr rhs, int op)
            : Node(out_type_str, name_str),
              m_op_str(op_str),
              m_lhs(lhs),
              m_rhs(rhs),
              m_op(op)
        {
        }

        bool isLinear(dim_t dims[4])
        {
            if (!m_set_is_linear) {
                m_linear = m_lhs->isLinear(dims) && m_rhs->isLinear(dims);
                m_set_is_linear = true;
            }
            return m_linear;
        }

        void genParams(std::stringstream &kerStream)
        {
            if (m_gen_param) return;
            if (!(m_lhs->isGenParam())) m_lhs->genParams(kerStream);
            if (!(m_rhs->isGenParam())) m_rhs->genParams(kerStream);
            m_gen_param = true;
        }

        int setArgs(cl::Kernel &ker, int id)
        {
            if (m_set_arg) return id;
            m_set_arg = true;

            id = m_lhs->setArgs(ker, id);
            id = m_rhs->setArgs(ker, id);
            return id;
        }

        void genOffsets(std::stringstream &kerStream, bool is_linear)
        {
            if (m_gen_offset) return;
            if (!(m_lhs->isGenOffset())) m_lhs->genOffsets(kerStream, is_linear);
            if (!(m_rhs->isGenOffset())) m_rhs->genOffsets(kerStream, is_linear);
            m_gen_offset = true;
        }

        void genKerName(std::stringstream &kerStream)
        {
            if (m_gen_name) return;
            m_lhs->genKerName(kerStream);
            m_rhs->genKerName(kerStream);

            // Make the dec representation of enum part of the Kernel name
            kerStream << "_" << std::setw(3) << std::setfill('0') << std::dec << m_op;
            kerStream << std::setw(3) << std::setfill('0') << std::dec << m_lhs->getId();
            kerStream << std::setw(3) << std::setfill('0') << std::dec << m_rhs->getId();
            kerStream << std::setw(3) << std::setfill('0') << std::dec << m_id << std::dec;
            m_gen_name = true;
        }

        void genFuncs(std::stringstream &kerStream)
        {
            if (m_gen_func) return;

            if (!(m_lhs->isGenFunc())) m_lhs->genFuncs(kerStream);
            if (!(m_rhs->isGenFunc())) m_rhs->genFuncs(kerStream);

            kerStream << m_type_str << " val" << m_id << " = "
                      << m_op_str << "(val" << m_lhs->getId()
                      << ", val" << m_rhs->getId() << ");"
                      << "\n";

            m_gen_func = true;
        }

        int setId(int id)
        {
            if (m_set_id) return id;

            id = m_lhs->setId(id);
            id = m_rhs->setId(id);

            m_id = id;
            m_set_id = true;

            return m_id + 1;
        }

        void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes)
        {
            if (m_set_id) return;

            m_lhs->getInfo(len, buf_count, bytes);
            m_rhs->getInfo(len, buf_count, bytes);
            len++;

            m_set_id = true;
            return;
        }

        void resetFlags()
        {
            if (m_set_id) {
                resetCommonFlags();
                m_lhs->resetFlags();
                m_rhs->resetFlags();
            }
        }
    };

}

}
