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

namespace cuda
{

namespace JIT
{

    static const int MAX_CHILDREN = 2;
    class Node;
    using std::shared_ptr;
    using std::vector;
    typedef shared_ptr<Node> Node_ptr;

    typedef struct
    {
        int id;
        std::array<int, MAX_CHILDREN> child_ids;
    } Node_ids;

    typedef std::unordered_map<Node *, int> Node_map_t;
    typedef Node_map_t::iterator Node_map_iter;

    class Node
    {
    protected:
        const std::string m_type_str;
        const std::string m_name_str;
        const int m_height;
        const std::array<Node_ptr, MAX_CHILDREN> m_children;

    public:

        Node(const char *type_str, const char *name_str, const int height,
             const std::array<Node_ptr, MAX_CHILDREN> children)
            : m_type_str(type_str),
              m_name_str(name_str),
              m_height(height),
              m_children(children)
        {}

        int getNodesMap(Node_map_t &node_map,
                        vector<Node *> &full_nodes,
                        vector<Node_ids> &full_ids)
        {
            auto iter = node_map.find(this);
            if (iter == node_map.end()) {
                Node_ids ids;
                for (int i = 0; i < MAX_CHILDREN && m_children[i] != nullptr; i++) {
                    ids.child_ids[i] = m_children[i]->getNodesMap(node_map, full_nodes, full_ids);
                }
                ids.id = node_map.size();
                node_map[this] = ids.id;
                full_nodes.push_back(this);
                full_ids.push_back(ids);
                return ids.id;
            }
            return iter->second;
        }

        virtual void genKerName (std::stringstream &kerStream, Node_ids ids) {}
        virtual void genParams  (std::stringstream &kerStream, int id, bool is_linear) {}
        virtual void genOffsets (std::stringstream &kerStream, int id, bool is_linear) {}
        virtual void genFuncs   (std::stringstream &kerStream, Node_ids) {}

        virtual void setArgs (std::vector<void *> &args, bool is_linear) { }

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
