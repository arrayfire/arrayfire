#pragma once
#include "Node.hpp"

namespace opencl
{

namespace JIT
{

    class ScalarNode : public Node
    {
    private:
        const cl_double2 m_val;
        const bool m_double;
        const bool m_complex;
        std::string m_name_str;
        bool m_gen_name;
        bool m_set_arg;

    public:

        ScalarNode(const double val, bool isDouble)
            : Node("float"),
              m_double(isDouble),
              m_complex(false),
              m_name_str("f"),
              m_gen_name(false),
              m_set_arg(false)
        {
            m_val = {val, 0};
            if (isDouble) {
                m_type_str = std::string("double");
                m_name_str = std::string("d");
            }
        }

        ScalarNode(const double2 val, bool isDouble)
            : Node("float2"),
              m_val(val),
              m_double(isDouble),
              m_complex(true),
              m_name_str("c"),
              m_gen_name(false),
              m_set_arg(false)
        {
            if (isDouble) {
                m_type_str = std::string("double2");
                m_name_str = std::string("z");
            }
        }

        void genKerName(std::stringstream &Stream, bool genInputs)
        {
            if (!genInputs) return;
            if (m_gen_name) return;

            Stream << m_name_str;
            m_gen_name = true;
        }

        void genParams(std::stringstream &Stream)
        {
            if (m_gen_param) return;
            Stream << m_type_str << " scalar" << m_id << ", " << std::endl;
            m_gen_param = true;
        }

        int setArgs(cl::Kernel &ker, int id)
        {
            if (m_set_arg) return id;

            if (!m_complex) {

                if (m_double) {
                    ker.setArg(id, (double)(val.s0));
                } else {
                    ker.setArg(id, (float)(val.s0));
                }

            } else {

                if (m_double) {
                    ker.setArg(id, (val));
                } else {
                    float2 valf = {(float)val.s0, (float)val.s1};
                    ker.setArg(id, valf);
                }

            }

            m_set_arg = true;
            return id + 1;
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
