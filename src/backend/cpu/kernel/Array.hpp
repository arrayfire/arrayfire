/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <common/jit/ModdimNode.hpp>
#include <common/jit/Node.hpp>
#include <common/jit/NodeIterator.hpp>
#include <jit/BufferNode.hpp>
#include <jit/Node.hpp>
#include <jit/UnaryNode.hpp>
#include <platform.hpp>
#include <vector>

namespace cpu {
namespace kernel {

/// Clones nodes and update the child pointers
std::vector<std::shared_ptr<common::Node>> cloneNodes(
    const std::vector<common::Node *> &nodes,
    const std::vector<common::Node_ids> &ids) {
    using common::Node;
    // find all moddims in the tree
    std::vector<std::shared_ptr<Node>> node_clones;
    node_clones.reserve(nodes.size());
    transform(begin(nodes), end(nodes), back_inserter(node_clones),
              [](Node *n) { return n->clone(); });

    for (common::Node_ids id : ids) {
        auto &children = node_clones[id.id]->m_children;
        for (int i = 0; i < Node::kMaxChildren && children[i] != nullptr; i++) {
            children[i] = node_clones[id.child_ids[i]];
        }
    }
    return node_clones;
}

/// Sets the shape of the buffer nodes under the moddims node to the new shape
void propagateModdimsShape(
    std::vector<std::shared_ptr<common::Node>> &node_clones) {
    using common::NodeIterator;
    for (auto &node : node_clones) {
        if (node->getOp() == af_moddims_t) {
            common::ModdimNode *mn =
                static_cast<common::ModdimNode *>(node.get());

            NodeIterator<> it(node.get());
            while (it != NodeIterator<>()) {
                it = find_if(it, NodeIterator<>(), common::isBuffer);
                if (it == NodeIterator<>()) { break; }

                it->setShape(mn->m_new_shape);

                ++it;
            }
        }
    }
}

/// Removes nodes whos operation matchs a unary operation \p op.
void removeNodeOfOperation(std::vector<std::shared_ptr<common::Node>> &nodes,
                           std::vector<common::Node_ids> &ids, af_op_t op) {
    using common::Node;

    std::vector<std::vector<std::shared_ptr<Node>>::iterator> moddims_loc;
    for (size_t nid = 0; nid < nodes.size(); nid++) {
        auto &node = nodes[nid];

        for (int i = 0;
             i < Node::kMaxChildren && node->m_children[i] != nullptr; i++) {
            if (node->m_children[i]->getOp() == op) {
                // replace moddims
                auto moddim_node    = node->m_children[i];
                node->m_children[i] = moddim_node->m_children[0];

                int parent_id = ids[nid].id;
                int moddim_id = ids[parent_id].child_ids[i];
                moddims_loc.emplace_back(begin(nodes) + moddim_id);
            }
        }
    }

    for (auto &loc : moddims_loc) { nodes.erase(loc); }
}

template<typename T>
void evalMultiple(std::vector<Param<T>> arrays,
                  std::vector<common::Node_ptr> output_nodes_) {
    using common::ModdimNode;
    using common::Node;
    using common::Node_map_t;
    using common::NodeIterator;

    af::dim4 odims = arrays[0].dims();
    af::dim4 ostrs = arrays[0].strides();

    Node_map_t nodes;
    std::vector<T *> ptrs;
    std::vector<TNode<T> *> output_nodes;
    std::vector<common::Node *> full_nodes;
    std::vector<common::Node_ids> ids;

    int narrays = static_cast<int>(arrays.size());
    for (int i = 0; i < narrays; i++) {
        ptrs.push_back(arrays[i].get());
        output_nodes_[i]->getNodesMap(nodes, full_nodes, ids);
    }

    auto node_clones = cloneNodes(full_nodes, ids);

    for (auto &n : output_nodes_) {
        if (n->getOp() == af_moddims_t) {
            // if the output node is a moddims node, then set the output node to
            // be the child of the moddims node. This is necessary because we
            // remove the moddim nodes from the tree later
            output_nodes.push_back(static_cast<TNode<T> *>(
                node_clones[nodes[n->m_children[0].get()]].get()));
        } else {
            output_nodes.push_back(
                static_cast<TNode<T> *>(node_clones[nodes[n.get()]].get()));
        }
    }

    propagateModdimsShape(node_clones);
    removeNodeOfOperation(node_clones, ids, af_moddims_t);

    bool is_linear = true;
    for (auto &node : node_clones) { is_linear &= node->isLinear(odims.get()); }

    int num_nodes        = node_clones.size();
    int num_output_nodes = output_nodes.size();
    if (is_linear) {
        int num = arrays[0].dims().elements();
        int cnum =
            jit::VECTOR_LENGTH * std::ceil(double(num) / jit::VECTOR_LENGTH);
        for (int i = 0; i < cnum; i += jit::VECTOR_LENGTH) {
            int lim = std::min(jit::VECTOR_LENGTH, num - i);
            for (int n = 0; n < num_nodes; n++) {
                node_clones[n]->calc(i, lim);
            }
            for (int n = 0; n < num_output_nodes; n++) {
                std::copy(output_nodes[n]->m_val.begin(),
                          output_nodes[n]->m_val.begin() + lim, ptrs[n] + i);
            }
        }
    } else {
        for (int w = 0; w < (int)odims[3]; w++) {
            dim_t offw = w * ostrs[3];

            for (int z = 0; z < (int)odims[2]; z++) {
                dim_t offz = z * ostrs[2] + offw;

                for (int y = 0; y < (int)odims[1]; y++) {
                    dim_t offy = y * ostrs[1] + offz;

                    int dim0  = odims[0];
                    int cdim0 = jit::VECTOR_LENGTH *
                                std::ceil(double(dim0) / jit::VECTOR_LENGTH);
                    for (int x = 0; x < (int)cdim0; x += jit::VECTOR_LENGTH) {
                        int lim  = std::min(jit::VECTOR_LENGTH, dim0 - x);
                        dim_t id = x + offy;

                        for (int n = 0; n < num_nodes; n++) {
                            node_clones[n]->calc(x, y, z, w, lim);
                        }
                        for (int n = 0; n < num_output_nodes; n++) {
                            std::copy(output_nodes[n]->m_val.begin(),
                                      output_nodes[n]->m_val.begin() + lim,
                                      ptrs[n] + id);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
