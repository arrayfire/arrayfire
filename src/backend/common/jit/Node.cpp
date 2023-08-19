/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <build_version.hpp>
#include <common/defines.hpp>
#include <common/deterministicHash.hpp>
#include <common/jit/Node.hpp>
#include <common/util.hpp>

#include <sstream>
#include <string>
#include <vector>

using std::vector;

namespace arrayfire {
namespace common {

int Node::getNodesMap(Node_map_t &node_map, vector<Node *> &full_nodes,
                      vector<Node_ids> &full_ids) {
    auto iter = node_map.find(this);
    if (iter == node_map.end()) {
        Node_ids ids{};

        for (int i = 0; i < kMaxChildren && m_children[i] != nullptr; i++) {
            ids.child_ids[i] =
                m_children[i]->getNodesMap(node_map, full_nodes, full_ids);
        }
        ids.id         = static_cast<int>(node_map.size());
        node_map[this] = ids.id;
        full_nodes.push_back(this);
        full_ids.push_back(ids);
        return ids.id;
    }
    return iter->second;
}

std::string getFuncName(const vector<Node *> &output_nodes,
                        const vector<Node *> &full_nodes,
                        const vector<Node_ids> &full_ids, const bool is_linear,
                        const bool loop0, const bool loop1, const bool loop2,
                        const bool loop3) {
    std::string funcName;
    funcName.reserve(512);
    funcName = (is_linear ? 'L' : 'G');
    funcName += (loop0 ? '0' : 'X');
    funcName += (loop1 ? '1' : 'X');
    funcName += (loop2 ? '2' : 'X');
    funcName += (loop3 ? '3' : 'X');

    for (const auto &node : output_nodes) {
        funcName += '_';
        funcName += node->getNameStr();
    }

    for (int i = 0; i < static_cast<int>(full_nodes.size()); i++) {
        full_nodes[i]->genKerName(funcName, full_ids[i]);
    }

    return "KER" + std::to_string(deterministicHash(funcName));
}

bool NodePtr_equalto::operator()(const Node *l, const Node *r) const noexcept {
    return *l == *r;
}

auto isBuffer(const Node &ptr) -> bool { return ptr.isBuffer(); }

auto isScalar(const Node &ptr) -> bool { return ptr.isScalar(); }

bool Node::isLinear(const dim_t dims[4]) const { return true; }

/// This function returns true if the \p node is a Shift node or a Buffer node
auto isBufferOrShift(const Node_ptr &node) -> bool {
    return node->getNodeType() == kNodeType::Buffer ||
           node->getNodeType() == kNodeType::Shift;
}

}  // namespace common
}  // namespace arrayfire

size_t std::hash<arrayfire::common::Node *>::operator()(
    arrayfire::common::Node *const node) const noexcept {
    arrayfire::common::Node *const node_ptr =
        static_cast<arrayfire::common::Node *const>(node);
    return node_ptr->getHash();
}
