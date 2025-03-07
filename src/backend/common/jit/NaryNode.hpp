/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <backend.hpp>
#include <common/defines.hpp>
#include <common/jit/Node.hpp>

#include <nonstd/span.hpp>
#include <array>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace arrayfire {
namespace common {

class NaryNode : public Node {
   private:
    int m_num_children;
    const char *m_op_str;

   protected:
    af_op_t m_op;

   public:
    NaryNode(const af::dtype type, const char *op_str, const int num_children,
             const std::array<common::Node_ptr, Node::kMaxChildren> &&children,
             const af_op_t op, const int height)
        : common::Node(
              type, height,
              std::forward<
                  const std::array<common::Node_ptr, Node::kMaxChildren>>(
                  children),
              kNodeType::Nary)
        , m_num_children(num_children)
        , m_op_str(op_str)
        , m_op(op) {
        static_assert(std::is_nothrow_move_assignable<NaryNode>::value,
                      "NaryNode is not move assignable");
        static_assert(std::is_nothrow_move_constructible<NaryNode>::value,
                      "NaryNode is not move constructible");
    }

    NaryNode(NaryNode &&other) noexcept = default;

    NaryNode(const NaryNode &other) = default;

    /// Default copy assignment operator
    NaryNode &operator=(const NaryNode &node) = default;

    /// Default move assignment operator
    NaryNode &operator=(NaryNode &&node) noexcept = default;

    void swap(NaryNode &other) noexcept {
        using std::swap;
        Node::swap(other);
        swap(m_num_children, other.m_num_children);
        swap(m_op_str, other.m_op_str);
        swap(m_op, other.m_op);
    }

    af_op_t getOp() const noexcept final { return m_op; }

    virtual std::unique_ptr<Node> clone() override {
        return std::make_unique<NaryNode>(*this);
    }

    void genKerName(std::string &kerString,
                    const common::Node_ids &ids) const final {
        // Make the dec representation of enum part of the Kernel name
        kerString += '_';
        kerString += std::to_string(m_op);
        kerString += ',';
        for (int i = 0; i < m_num_children; i++) {
            kerString += std::to_string(ids.child_ids[i]);
            kerString += ',';
        }
        kerString += std::to_string(ids.id);
    }

    void genFuncs(std::stringstream &kerStream,
                  const common::Node_ids &ids) const final {
        kerStream << getTypeStr() << " val" << ids.id << " = " << m_op_str
                  << "(";
        for (int i = 0; i < m_num_children; i++) {
            if (i > 0) kerStream << ", ";
            kerStream << "val" << ids.child_ids[i];
        }
        kerStream << ");\n";
    }
};

template<typename Ti, int N, typename FUNC>
common::Node_ptr createNaryNode(
    const af::dim4 &odims, FUNC createNode,
    std::array<const detail::Array<Ti> *, N> &&children) {
    std::array<common::Node_ptr, N> childNodes;
    std::array<common::Node *, N> nodes;
    for (int i = 0; i < N; i++) {
        childNodes[i] = move(children[i]->getNode());
        nodes[i]      = childNodes[i].get();
    }

    common::Node_ptr ptr = createNode(childNodes);

    switch (detail::passesJitHeuristics<Ti>(nodes)) {
        case kJITHeuristics::Pass: {
            return ptr;
        }
        case kJITHeuristics::TreeHeight:
        case kJITHeuristics::KernelParameterSize: {
            int max_height_index = 0;
            int max_height       = 0;
            for (int i = 0; i < N; i++) {
                if (max_height < childNodes[i]->getHeight()) {
                    max_height_index = i;
                    max_height       = childNodes[i]->getHeight();
                }
            }
            children[max_height_index]->eval();
            return createNaryNode<Ti, N>(odims, createNode, move(children));
        }
        case kJITHeuristics::MemoryPressure: {
            for (auto &c : children) { c->eval(); }  // TODO: use evalMultiple()
            return ptr;
        }
    }
    return ptr;
}
}  // namespace common
}  // namespace arrayfire
