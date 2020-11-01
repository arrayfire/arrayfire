/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/defines.hpp>
#include <common/jit/Node.hpp>
#include <common/util.hpp>

#include <sstream>
#include <string>
#include <vector>

using std::vector;

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
        ids.id         = node_map.size();
        node_map[this] = ids.id;
        full_nodes.push_back(this);
        full_ids.push_back(ids);
        return ids.id;
    }
    return iter->second;
}

std::string getFuncName(const vector<Node *> &output_nodes,
                        const vector<Node *> &full_nodes,
                        const vector<Node_ids> &full_ids, bool is_linear) {
    std::string funcName;
    funcName.reserve(512);
    funcName = is_linear ? 'L' : 'G';

    for (const auto &node : output_nodes) { funcName += node->getNameStr(); }

    for (int i = 0; i < static_cast<int>(full_nodes.size()); i++) {
        full_nodes[i]->genKerName(funcName, full_ids[i]);
    }

    return "KER" + std::to_string(deterministicHash(funcName));
}

}  // namespace common
