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
        Node *m_lhs, *m_rhs;
        int m_op;

    public:
        BinaryNode(const char *out_type_str,
                   const char *op_str,
                   Node *lhs, Node *rhs, int op)
            : Node(out_type_str),
              m_op_str(op_str),
              m_lhs(lhs),
              m_rhs(rhs),
              m_op(op)
        {
        }

        void genParams(std::stringstream &Stream)
        {
            if (m_gen_param) return;
            if (!(m_lhs->isGenParam())) m_lhs->genParams(Stream);
            if (!(m_rhs->isGenParam())) m_rhs->genParams(Stream);
            m_gen_param = true;
        }

        void genOffsets(std::stringstream &Stream)
        {
            if (m_gen_offset) return;
            if (!(m_lhs->isGenOffset())) m_lhs->genOffsets(Stream);
            if (!(m_rhs->isGenOffset())) m_rhs->genOffsets(Stream);
            m_gen_offset = true;
        }

        void genKerName(std::stringstream &Stream, bool genInputs)
        {
            if (!genInputs) {
                // Make the hex representation of enum part of the Kernel name
                Stream << std::setw(2) << std::setfill('0') << std::hex << m_op << std::dec;
            }
            m_lhs->genKerName(Stream, genInputs);
            m_rhs->genKerName(Stream, genInputs);
        }

        void genFuncs(std::stringstream &Stream)
        {
            if (m_gen_func) return;

            if (!(m_lhs->isGenFunc())) m_lhs->genFuncs(Stream);
            if (!(m_rhs->isGenFunc())) m_rhs->genFuncs(Stream);

            Stream << m_type_str << " val" << m_id << " = "
                   << m_op_str << "(val" << m_lhs->getId()
                   << ", val" << m_rhs->getId() << ");"
                   << std::endl;

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

    };

}

}
