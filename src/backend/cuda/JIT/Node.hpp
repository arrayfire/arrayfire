/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/array.h>
#include <optypes.hpp>
#include <string>
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>

namespace cuda
{

namespace JIT
{
    typedef std::map<std::string, bool> str_map_t;
    typedef str_map_t::iterator str_map_iter;
    using boost::shared_ptr;

    class Node
    {
    protected:
        std::string m_type_str;
        int m_id;
        bool m_set_id;
        bool m_gen_func;
        bool m_gen_param;
        bool m_gen_offset;
        bool m_set_arg;

    public:

        Node(const char *type_str)
            : m_type_str(type_str), m_id(-1),
              m_set_id(false),
              m_gen_func(false),
              m_gen_param(false),
              m_gen_offset(false),
              m_set_arg(false)
        {}

        virtual void genKerName(std::stringstream &kerStream, bool genInputs) {}
        virtual void genParams  (std::stringstream &kerStream, std::stringstream &annStream) {}
        virtual void genOffsets (std::stringstream &kerStream) {}
        virtual void genFuncs   (std::stringstream &kerStream, str_map_t &declStrs)
        { m_gen_func = true;}

        virtual int setId(int id) { m_set_id = true; return id; }
        virtual void setArgs(std::vector<void *> &args) { m_set_arg = true; }

        virtual void resetFlags() {}

        std::string getTypeStr() { return m_type_str; }

        bool isGenFunc() { return m_gen_func; }
        bool isGenParam() { return m_gen_param; }
        bool isGenOffset() { return m_gen_offset; }

        int getId()  { return m_id; }

        virtual ~Node() {}
    };

    typedef shared_ptr<Node> Node_ptr;

}

}
