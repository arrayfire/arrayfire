/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <optypes.hpp>

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

namespace cuda
{

namespace JIT
{
    class Node;

    typedef struct
    {
        int id;
        std::vector<int> child_ids;
    } Node_ids;

    typedef std::unordered_map<std::string, bool> str_map_t;
    typedef str_map_t::iterator str_map_iter;
    typedef std::shared_ptr<Node> Node_ptr;
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
        virtual void genParams  (std::stringstream &kerStream,
                                 std::stringstream &annStream,
                                 int id, bool is_linear) {}
        virtual void genOffsets (std::stringstream &kerStream, int id, bool is_linear) {}
        virtual void genFuncs   (std::stringstream &kerStream, str_map_t &declStrs,
                                 Node_ids id, bool is_linear)
        {}

        virtual void setArgs(std::vector<void *> &args, bool is_linear) {}
        virtual bool isLinear(dim_t dims[4]) { return true; }

        virtual void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes)
        {
            len++;
        }

        virtual bool isBuffer() { return false; }

        std::string getTypeStr() { return m_type_str; }

        int getHeight()  { return m_height; }
        std::string getNameStr() { return m_name_str; }

        virtual ~Node() {}
    };
}

}
