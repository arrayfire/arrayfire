/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <JIT/Node.hpp>

#include <spdlog/fmt/ostr.h>

#include <string>
#include <vector>

using std::vector;

namespace opencl {
namespace JIT {

int Node::getNodesMap(Node_map_t &node_map,
                      vector<const Node *> &full_nodes,
                      vector<Node_ids> &full_ids) const
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

void Node::genKerName(std::stringstream &kerStream, Node_ids ids) const
{
    fmt::print(kerStream, "_{0}{1:0<3}", m_name_str, ids.id);
}

}  // JIT
}  // opencl
