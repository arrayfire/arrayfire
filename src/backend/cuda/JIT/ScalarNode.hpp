#pragma once
#include "Node.hpp"
#include <math.hpp>

namespace cuda
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
              m_val(scalar<cdouble>(val)),
              m_double(isDouble),
              m_complex(false),
              m_name_str("f"),
              m_gen_name(false),
              m_set_arg(false)
        {
            if (isDouble) {
                m_type_str = std::string("double");
                m_name_str = std::string("d");
            }
        }

        ScalarNode(const cl_double2 val, bool isDouble)
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

        void genKerName(std::stringstream &kerStream, bool genInputs)
        {
            if (!genInputs) return;
            if (m_gen_name) return;

            kerStream << m_name_str;
            m_gen_name = true;
        }

        void genParams(std::stringstream &kerStream)
        {
            if (m_gen_param) return;
            kerStream << m_type_str << " %val" << m_id << ", " << std::endl;
            m_gen_param = true;
        }

        void genOffsets(std::stringstream &kerStream)
        {
            if (m_gen_offset) return;
            m_gen_offset = true;
        }

        void genFuncs(std::stringstream &kerStream, std::stringstream &declStream)
        {
            if (m_gen_func) return;
            m_gen_func = true;
        }

        int setId(int id)
        {
            if (m_set_id) return id;

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
            m_gen_name = false;
            m_set_arg = false;
        }
    };

}

}
