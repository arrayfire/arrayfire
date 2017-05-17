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
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace opencl
{

namespace JIT
{

    class Node;
    using std::shared_ptr;
    typedef shared_ptr<Node> Node_ptr;

    typedef struct
    {
        int id;
        std::vector<int> child_ids;
    } Node_ids;

    typedef std::unordered_map<Node *, Node_ids> Node_map_t;
    typedef Node_map_t::iterator Node_map_iter;

    class Node
    {
    protected:
        const std::string m_type_str;
        const std::string m_name_str;
        const int m_height;
        const std::vector<Node_ptr> m_children;

    public:

        Node(const char *type_str, const char *name_str, const int height,
             const std::vector<Node_ptr> children)
            : m_type_str(type_str),
              m_name_str(name_str),
              m_height(height),
              m_children(children)
        {}

        void getNodesMap(Node_map_t &node_map)
        {
            if (node_map.find(this) == node_map.end()) {
                Node_ids ids;
                for (const auto &child : m_children) {
                    child->getNodesMap(node_map);
                    ids.child_ids.push_back(node_map[child.get()].id);
                }
                ids.id = node_map.size();
                node_map[this] = ids;
            }
        }

        virtual void genKerName(std::stringstream &kerStream, Node_ids ids) {}
        virtual void genParams  (std::stringstream &kerStream, int id, bool is_linear) {}
        virtual void genOffsets (std::stringstream &kerStream, int id, bool is_linear) {}
        virtual void genFuncs   (std::stringstream &kerStream, Node_ids) {}

        virtual int setArgs (cl::Kernel &ker, int id, bool is_linear) { return id; }

        virtual void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes)
        {
            len++;
        }

        virtual bool isBuffer() { return false; }
        virtual bool isLinear(dim_t dims[4]) { return true; }
        std::string getTypeStr() { return m_type_str; }
        int getHeight()  { return m_height; }
        std::string getNameStr() { return m_name_str; }

        virtual ~Node() {}
    };
}

}
