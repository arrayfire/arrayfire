#pragma once
#include "Node.hpp"

namespace opencl
{

namespace JIT
{

    class BufferNode : public Node
    {
    private:
        const Param m_param;
        std::string m_name_str;
        std::string m_info_str;

    public:

        BufferNode(const char *type_str, const Param param)
            : Node(type_str),
              m_param(param),
              m_name_str(),
              m_info_str()
        {}

        void genFuncName(std::stringstream &Stream)
        {
            if (m_gen_name) return;
            Stream << "_"  << m_type_str << "Array_";
            m_gen_name = true;
        }

        void genParams(std::stringstream &Stream)
        {
            if (m_gen_param) return;
            Stream << "__global " << m_type_str << " *in" << m_id
                   << ", KParam iInfo" << m_id << ", " << std::endl;
            m_gen_param = true;
        }

        void genOffsets(std::stringstream &Stream)
        {
            if (m_gen_offset) return;

            std::string name_str = std::string("int idx") + std::to_string(m_id);
            std::string info_str = std::string("iInfo") + std::to_string(m_id);;

            Stream << name_str << " = "
                   << info_str << ".strides[3] * id3 + "
                   << info_str << ".strides[2] * id2 + "
                   << info_str << ".strides[1] * id1 + "
                   << "id0 + " << info_str << ".offset;"
                   << std::endl;

            m_gen_offset = true;
        }

        void genFuncs(std::stringstream &Stream)
        {
            if (m_gen_func) return;

            Stream << m_type_str << " val" << m_id << " = "
                   << "in" << m_id << "[idx" << m_id << "];"
                   << std::endl;

            m_gen_func = true;
        }

        int setId(int id)
        {
            if (m_set_id) return id;

            m_id = id;
            m_set_id = true;

            return m_id + 1;
        }

    };

}

}
