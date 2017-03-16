/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
#include <platform.hpp>
#include <TNJ/Node.hpp>
#include <vector>

namespace cpu
{
namespace kernel
{

template<typename T>
void evalMultiple(std::vector<Array<T>> arrays)
{
    af::dim4 odims = arrays[0].dims();
    af::dim4 ostrs = arrays[0].strides();

    int devId = cpu::getActiveDeviceId();
    TNJ::Node_map_t nodes;
    std::vector<T *> ptrs;
    std::vector<TNJ::TNode<T> *> output_nodes;

    for (auto &arr : arrays) {
        arr.setId(devId);
        ptrs.push_back(arr.data.get());
        output_nodes.push_back(reinterpret_cast<TNJ::TNode<T> *>(arr.node.get()));
        arr.node->getNodesMap(nodes);
    }

    bool is_linear = true;
    std::vector<TNJ::Node *> full_nodes(nodes.size());
    for(const auto &map_entry : nodes) {
        full_nodes[map_entry.second] = map_entry.first;
        is_linear &= map_entry.first->isLinear(odims.get());
    }

    if (is_linear) {
        int num = arrays[0].elements();
        for (int i = 0; i < num; i++) {
            for (int n = 0; n < (int)full_nodes.size(); n++) {
                full_nodes[n]->calc(i);
            }
            for (int n = 0; n < (int)output_nodes.size(); n++) {
                ptrs[n][i] = output_nodes[n]->m_val;
            }
        }
    } else {
        for (int w = 0; w < (int)odims[3]; w++) {
            dim_t offw = w * ostrs[3];

            for (int z = 0; z < (int)odims[2]; z++) {
                dim_t offz = z * ostrs[2] + offw;

                for (int y = 0; y < (int)odims[1]; y++) {
                    dim_t offy = y * ostrs[1] + offz;

                    for (int x = 0; x < (int)odims[0]; x++) {
                        dim_t id = x + offy;

                        for (int n = 0; n < (int)full_nodes.size(); n++) {
                            full_nodes[n]->calc(x, y, z, w);
                        }
                        for (int n = 0; n < (int)output_nodes.size(); n++) {
                            ptrs[n][id] = output_nodes[n]->m_val;
                        }
                    }
                }
            }
        }
    }
}

template<typename T>
void evalArray(Array<T> arr)
{
    arr.setId(cpu::getActiveDeviceId());
    T *ptr = arr.data.get();

    af::dim4 odims = arr.dims();
    af::dim4 ostrs = arr.strides();

    TNJ::Node_map_t nodes;
    arr.node->getNodesMap(nodes);

    bool is_linear = true;
    std::vector<TNJ::Node *> full_nodes(nodes.size());

    for(const auto &map_entry : nodes) {
        full_nodes[map_entry.second] = map_entry.first;
        is_linear &= map_entry.first->isLinear(odims.get());
    }

    TNJ::TNode<T> *output_node = reinterpret_cast<TNJ::TNode<T> *>(full_nodes.back());
    if (is_linear) {
        int num = arr.elements();
        for (int i = 0; i < num; i++) {
            for (int n = 0; n < (int)full_nodes.size(); n++) {
                full_nodes[n]->calc(i);
            }
            ptr[i] = output_node->m_val;
        }
    } else {
        for (int w = 0; w < (int)odims[3]; w++) {
            dim_t offw = w * ostrs[3];

            for (int z = 0; z < (int)odims[2]; z++) {
                dim_t offz = z * ostrs[2] + offw;

                for (int y = 0; y < (int)odims[1]; y++) {
                    dim_t offy = y * ostrs[1] + offz;

                    for (int x = 0; x < (int)odims[0]; x++) {
                        dim_t id = x + offy;

                        for (int n = 0; n < (int)full_nodes.size(); n++) {
                            full_nodes[n]->calc(x, y, z, w);
                        }
                        ptr[id] = output_node->m_val;
                    }
                }
            }
        }
    }
}

}
}
