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
#include <optypes.hpp>
#include <array>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace common {
    class NodeIterator;
}
using std::shared_ptr;
using std::vector;

namespace cuda
{

namespace JIT
{

    constexpr int MAX_CHILDREN = 3;
    class Node;

    typedef struct
    {
        int id;
        std::array<int, MAX_CHILDREN> child_ids;
    } Node_ids;

    using Node_ptr = shared_ptr<Node>;
    using Node_map_t = std::unordered_map<const Node *, int> ;
    using Node_map_iter = Node_map_t::iterator;

    class Node
    {
    protected:
        const std::string m_type_str;
        const std::string m_name_str;
        const std::array<Node_ptr, MAX_CHILDREN> m_children;
        const int m_height;
        friend class common::NodeIterator;

    public:

        Node(const char *type_str, const char *name_str, const int height,
             const std::array<Node_ptr, MAX_CHILDREN> children)
            : m_type_str(type_str),
              m_name_str(name_str),
              m_children(children),
              m_height(height) {}

        int getNodesMap(Node_map_t &node_map,
                        vector<const Node *> &full_nodes,
                        vector<Node_ids> &full_ids) const;

        virtual void genKerName (std::stringstream &kerStream, Node_ids ids) const {}
        virtual void genParams  (std::stringstream &kerStream, int id, bool is_linear) const {}
        virtual void genOffsets (std::stringstream &kerStream, int id, bool is_linear) const {}
        virtual void genFuncs   (std::stringstream &kerStream, Node_ids) const {}

        virtual void setArgs (std::vector<void *> &args, bool is_linear) const { }

        virtual void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes) const {
            len++;
        }

        // Return the size of the parameter in bytes that will be passed to the
        // kernel
        virtual short getParamBytes() const {
            return 0;
        }

        // Return the size of the size of the buffer node in bytes. Zero otherwise
        virtual size_t getBytes() const { return 0; }
        virtual bool isBuffer() const { return false; }
        virtual bool isLinear(dim_t dims[4]) const { return true; }
        std::string getTypeStr() const { return m_type_str; }
        int getHeight()  const { return m_height; }
        std::string getNameStr() const { return m_name_str; }

        virtual ~Node() {}
    };
}

}
