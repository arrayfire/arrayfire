/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <backend.hpp>

#include <cstddef>
#include <iterator>
#include <vector>

namespace arrayfire {
namespace common {

/// A node iterator that performs a breadth first traversal of the node tree
template<typename Node = common::Node>
class NodeIterator : public std::iterator<std::input_iterator_tag, Node> {
   public:
    using pointer   = Node*;
    using reference = Node&;

   private:
    std::vector<pointer> tree;
    size_t index = 0;

    /// Copies the children of the \p n Node to the end of the tree vector
    void copy_children_to_end(Node* n) {
        for (int i = 0; i < Node::kMaxChildren && n->m_children[i] != nullptr;
             i++) {
            auto ptr = n->m_children[i].get();
            if (find(begin(tree), end(tree), ptr) == end(tree)) {
                tree.push_back(ptr);
            }
        }
    }

   public:
    /// NodeIterator Constructor
    ///
    /// \param[in] root The root node of the tree
    NodeIterator(pointer root) : tree{root}, index(0) {
        tree.reserve(root->getHeight() * 8);
    }

    /// The equality operator
    ///
    /// \param[in] other the rhs of the node
    bool operator==(const NodeIterator& other) const noexcept {
        // If the tree vector is empty in the other iterator then this means
        // that the other iterator is a sentinel(end) node.
        if (other.tree.empty()) {
            // If the index is the same as the tree size then the index is past
            // the end of the tree
            return index == tree.size();
        }
        return index == other.index && tree == other.tree;
    }

    bool operator!=(const NodeIterator& other) const noexcept {
        return !operator==(other);
    }

    /// Advances the iterator by one node in the tree
    NodeIterator& operator++() noexcept {
        if (index < tree.size()) { copy_children_to_end(tree[index]); }
        index++;
        return *this;
    }

    /// @copydoc operator++()
    NodeIterator operator++(int) noexcept {
        NodeIterator before(*this);
        operator++();
        return before;
    }

    /// Advances the iterator by count nodes
    NodeIterator& operator+=(std::size_t count) noexcept {
        while (count-- > 0) { operator++(); }
        return *this;
    }

    reference operator*() const noexcept { return *tree[index]; }

    pointer operator->() const noexcept { return tree[index]; }

    /// Creates a sentinel iterator. This is equivalent to the end iterator
    NodeIterator()                                         = default;
    NodeIterator(const NodeIterator& other)                = default;
    NodeIterator(NodeIterator&& other) noexcept            = default;
    ~NodeIterator() noexcept                               = default;
    NodeIterator& operator=(const NodeIterator& other)     = default;
    NodeIterator& operator=(NodeIterator&& other) noexcept = default;
};

}  // namespace common
}  // namespace arrayfire
