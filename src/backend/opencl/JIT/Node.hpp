#pragma once
#include <af/array.h>
#include <cl.hpp>
#include <optypes.hpp>
#include <string>

namespace opencl
{

namespace JIT
{

    class Node
    {
    protected:
        std::string m_type_str;
        int m_id;
        bool m_set_id;
        bool m_gen_func;
        bool m_gen_param;
        bool m_gen_offset;

    public:

        Node(const char *type_str)
            : m_type_str(type_str), m_id(-1),
              m_set_id(false),
              m_gen_func(false),
              m_gen_param(false),
              m_gen_offset(false)
        {}

        virtual void genKerName(std::stringstream &Stream, bool genInputs) {}
        virtual void genParams  (std::stringstream &Stream) {}
        virtual void genOffsets (std::stringstream &Stream) {}
        virtual void genFuncs   (std::stringstream &Stream) { m_gen_func = true;}
        virtual int setId(int id) { m_set_id = true; return id; }
        std::string getTypeStr() { return m_type_str; }

        bool isGenFunc() { return m_gen_func; }
        bool isGenParam() { return m_gen_param; }
        bool isGenOffset() { return m_gen_offset; }

        int getId()  { return m_id; }

        virtual ~Node() {}
    };

}

}
