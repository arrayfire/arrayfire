/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <platform.hpp>
#include <af/array.h>
#include <optypes.hpp>
#include <string>
#include <vector>
#include <memory>

namespace opencl
{

namespace JIT
{
    using std::shared_ptr;

    class Node
    {
    protected:
        std::string m_type_str;
        std::string m_name_str;
        int m_id;
        bool m_set_id;
        bool m_gen_func;
        bool m_gen_param;
        bool m_gen_offset;
        bool m_gen_name;

    public:

        Node(const char *type_str, const char *name_str)
            : m_type_str(type_str),
              m_name_str(name_str),
              m_id(-1),
              m_set_id(false),
              m_gen_func(false),
              m_gen_param(false),
              m_gen_offset(false),
              m_gen_name(false)
        {}

        virtual void genKerName(std::stringstream &kerStream) {}
        virtual void genParams  (std::stringstream &kerStream) {}
        virtual void genOffsets (std::stringstream &kerStream, bool is_linear) {}
        virtual void genFuncs   (std::stringstream &kerStream) { m_gen_func = true;}

        virtual int setArgs (cl::Kernel &ker, int id) { return id; }

        virtual int setId(int id) { m_set_id = true; return id; }

        virtual void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes)
        {
            len = 0;
            buf_count = 0;
            bytes = 0;
        }


        virtual void resetFlags() {}

        virtual bool isLinear(dim_t dims[4]) { return true; }

        std::string getTypeStr() { return m_type_str; }

        bool isGenFunc() { return m_gen_func; }
        bool isGenParam() { return m_gen_param; }
        bool isGenOffset() { return m_gen_offset; }

        int getId()  { return m_id; }
        std::string getNameStr() { return m_name_str; }

        virtual ~Node() {}
    };

    typedef shared_ptr<Node> Node_ptr;

}

}
