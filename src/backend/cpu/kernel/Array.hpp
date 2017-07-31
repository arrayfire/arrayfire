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
    std::vector<TNJ::Node *> full_nodes;

    for (auto &arr : arrays) {
        arr.setId(devId);
        ptrs.push_back(arr.data.get());
        output_nodes.push_back(reinterpret_cast<TNJ::TNode<T> *>(arr.node.get()));
        arr.node->getNodesMap(nodes, full_nodes);
    }

    bool is_linear = true;
    for(auto node : full_nodes) {
        is_linear &= node->isLinear(odims.get());
    }

    if (is_linear) {
        int num = arrays[0].dims().elements();
        int cnum = TNJ::VECTOR_LENGTH * std::ceil(double(num) / TNJ::VECTOR_LENGTH);
        for (int i = 0; i < cnum; i += TNJ::VECTOR_LENGTH) {
            int lim = std::min(TNJ::VECTOR_LENGTH, num - i);
            for (int n = 0; n < (int)full_nodes.size(); n++) {
                full_nodes[n]->calc(i, lim);
            }
            for (int n = 0; n < (int)output_nodes.size(); n++) {
                std::copy(output_nodes[n]->m_val.begin(),
                          output_nodes[n]->m_val.begin() + lim,
                          ptrs[n] + i);
            }
        }
    } else {

        for (int w = 0; w < (int)odims[3]; w++) {
            dim_t offw = w * ostrs[3];

            for (int z = 0; z < (int)odims[2]; z++) {
                dim_t offz = z * ostrs[2] + offw;

                for (int y = 0; y < (int)odims[1]; y++) {
                    dim_t offy = y * ostrs[1] + offz;

                    int dim0 = odims[0];
                    int cdim0 = TNJ::VECTOR_LENGTH * std::ceil(double(dim0) / TNJ::VECTOR_LENGTH);
                    for (int x = 0; x < (int)cdim0; x += TNJ::VECTOR_LENGTH) {
                        int lim = std::min(TNJ::VECTOR_LENGTH, dim0 - x);
                        dim_t id = x + offy;

                        for (int n = 0; n < (int)full_nodes.size(); n++) {
                            full_nodes[n]->calc(x, y, z, w, lim);
                        }
                        for (int n = 0; n < (int)output_nodes.size(); n++) {
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

template<typename T>
void evalArray(Array<T> arr)
{
    evalMultiple<T>({arr});
}

}
}
