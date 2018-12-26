/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <common/defines.hpp>
#include <optypes.hpp>

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

namespace common {
template<typename T>
class NodeIterator;
}

namespace cpu {

namespace jit {
class Node;
constexpr int VECTOR_LENGTH = 256;

using Node_ptr      = std::shared_ptr<Node>;
using Node_map_t    = std::unordered_map<Node *, int>;
using Node_map_iter = Node_map_t::iterator;

template<typename T>
using array = std::array<T, VECTOR_LENGTH>;

class Node {
   public:
    static const int kMaxChildren = 2;

   protected:
    const int m_height;
    const std::array<Node_ptr, kMaxChildren> m_children;
    template<typename T>
    friend class common::NodeIterator;

   public:
    Node(const int height, const std::array<Node_ptr, kMaxChildren> children)
        : m_height(height), m_children(children) {}

    int getNodesMap(Node_map_t &node_map, std::vector<Node *> &full_nodes) {
        auto iter = node_map.find(this);
        if (iter == node_map.end()) {
            for (auto &child : m_children) {
                if (child == nullptr) break;
                child->getNodesMap(node_map, full_nodes);
            }
            int id         = static_cast<int>(node_map.size());
            node_map[this] = id;
            full_nodes.push_back(this);
            return id;
        }
        return iter->second;
    }

    int getHeight() { return m_height; }

    virtual void calc(int x, int y, int z, int w, int lim) {
        UNUSED(x);
        UNUSED(y);
        UNUSED(z);
        UNUSED(w);
        UNUSED(lim);
    }

    virtual void calc(int idx, int lim) {
        UNUSED(idx);
        UNUSED(lim);
    }

    virtual void getInfo(unsigned &len, unsigned &buf_count,
                         unsigned &bytes) const {
        UNUSED(buf_count);
        UNUSED(bytes);
        len++;
    }

    virtual bool isLinear(const dim_t *dims) const {
        UNUSED(dims);
        return true;
    }
    virtual bool isBuffer() const { return false; }
    virtual ~Node() {}

    virtual size_t getBytes() const { return 0; }
};

template<typename T>
class TNode : public Node {
   public:
    alignas(16) jit::array<T> m_val;

   public:
    TNode(T val, const int height,
          const std::array<Node_ptr, kMaxChildren> children)
        : Node(height, children) {
        m_val.fill(val);
    }
};

template<typename T>
using TNode_ptr = std::shared_ptr<TNode<T>>;
}  // namespace jit

}  // namespace cpu
