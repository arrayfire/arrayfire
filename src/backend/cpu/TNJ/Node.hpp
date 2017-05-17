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
#include <vector>
#include <memory>
#include <unordered_map>

namespace cpu
{

namespace TNJ
{

    class Node;
    using std::shared_ptr;
    typedef shared_ptr<Node> Node_ptr;

    typedef std::unordered_map<Node *, int> Node_map_t;
    typedef Node_map_t::iterator Node_map_iter;

    class Node
    {

    protected:

        const int m_height;
        const std::vector<Node_ptr> m_children;

    public:
        Node(const int height, const std::vector<Node_ptr> children) :
            m_height(height),
            m_children(children)
        {}

        void getNodesMap(Node_map_t &node_map)
        {
            if (node_map.find(this) == node_map.end()) {
                for (const auto &child : m_children) {
                    child->getNodesMap(node_map);
                }
                int id = node_map.size();
                node_map[this] = id;
            }
        }

        int getHeight() { return m_height; }

        virtual void calc(int x, int y, int z, int w)
        {
        }

        virtual void calc(int idx)
        {
        }

        virtual void getInfo(unsigned &len, unsigned &buf_count, unsigned &bytes)
        {
            len++;
        }

        virtual bool isLinear(const dim_t *dims) { return true; }
        virtual bool isBuffer() { return false; }
        virtual ~Node() {}

    };

    template<typename T>
    class TNode : public Node
    {
    public:
        T m_val;
    public:
        TNode(T val, const int height, const std::vector<Node_ptr> children) :
            Node(height, children),
            m_val(val)
            {
            }
    };

    template<typename T>
    using TNode_ptr = std::shared_ptr<TNode<T>>;
}

}
