#pragma once
#include "Node.hpp"

namespace opencl
{

namespace JIT
{

    class ScalarNode : public Node
    {
    private:
        const double m_val;
        const bool m_double;
        std::string m_name_str;
        std::string m_info_str;

    public:

        ScalarNode(const char *type_str, const double val, bool isDouble)
            : Node(type_str),
              m_val(val),
              m_double(isDouble),
              m_name_str(),
              m_info_str()
        {}

        void genFuncName(std::stringstream &Stream)
        {
            if (m_gen_name) return;
            Stream << "_"  << m_type_str << "Scalar_";
            m_gen_name = true;
        }

        void genParams(std::stringstream &Stream)
        {
            if (m_gen_param) return;
            Stream << m_type_str << " scalar" << m_id << ", " << std::endl;
            m_gen_param = true;
        }

        void genOffsets(std::stringstream &Stream)
        {
            if (m_gen_offset) return;
            m_gen_offset = true;
        }

        void genFuncs(std::stringstream &Stream)
        {
            if (m_gen_func) return;

            Stream << m_type_str << " val" << m_id << " = "
                   << "scalar" << m_id << ";"
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
