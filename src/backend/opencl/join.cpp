/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <err_opencl.hpp>
#include <join.hpp>
#include <kernel/memcopy.hpp>

#include <algorithm>
#include <cassert>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

using af::dim4;
using arrayfire::common::half;
using arrayfire::common::Node;
using arrayfire::common::Node_ptr;
using std::vector;

namespace arrayfire {
namespace opencl {
template<typename T>
Array<T> join(const int jdim, const Array<T> &first, const Array<T> &second) {
    const dim4 &fdims{first.dims()};
    const dim4 &sdims{second.dims()};
    // All dimensions except join dimension must be equal
    assert((jdim == 0 ? true : fdims.dims[0] == sdims.dims[0]) &&
           (jdim == 1 ? true : fdims.dims[1] == sdims.dims[1]) &&
           (jdim == 2 ? true : fdims.dims[2] == sdims.dims[2]) &&
           (jdim == 3 ? true : fdims.dims[3] == sdims.dims[3]));

    // Compute output dims
    dim4 odims(fdims);
    odims.dims[jdim] += sdims.dims[jdim];
    Array<T> out = createEmptyArray<T>(odims);

    // topspeed is achieved when byte size(in+out) ~= L2CacheSize
    //
    // 1 array: memcpy always copies 1 array.  topspeed
    //      --> size(in) <= L2CacheSize/2
    // 2 arrays: topspeeds
    //      - size(in) < L2CacheSize/2/2
    //          --> JIT can copy 2 arrays in // and is fastest
    //              (condition: array sizes have to be identical)
    //      - size(in) < L2CacheSize/2
    //          --> memcpy will achieve highest speed, although the kernel
    //              has to be called twice
    //      - size(in) >= L2CacheSize/2
    //          --> memcpy will achieve veryLargeArray speed.  The kernel
    //              will be called twice
    if (fdims.dims[jdim] == sdims.dims[jdim]) {
        const size_t L2CacheSize{getL2CacheSize(opencl::getDevice())};
        if (!(first.isReady() || second.isReady()) ||
            (fdims.elements() * sizeof(T) * 2 * 2 < L2CacheSize)) {
            // Both arrays have same size & everything fits into the cache,
            // so thread in 1 JIT kernel, iso individual copies which is
            // always slower
            const dim4 &outStrides{out.strides()};
            vector<Param> outputs{
                {out.get(),
                 {{fdims.dims[0], fdims.dims[1], fdims.dims[2], fdims.dims[3]},
                  {outStrides.dims[0], outStrides.dims[1], outStrides.dims[2],
                   outStrides.dims[3]},
                  0}},
                {out.get(),
                 {{sdims.dims[0], sdims.dims[1], sdims.dims[2], sdims.dims[3]},
                  {outStrides.dims[0], outStrides.dims[1], outStrides.dims[2],
                   outStrides.dims[3]},
                  fdims.dims[jdim] * outStrides.dims[jdim]}}};
            // Extend the life of the returned node, bij saving the
            // corresponding shared_ptr
            const Node_ptr fNode{first.getNode()};
            const Node_ptr sNode{second.getNode()};
            vector<Node *> nodes{fNode.get(), sNode.get()};
            evalNodes(outputs, nodes);
            return out;
        }
        // continue because individually processing is faster
    }

    // Handle each array individually
    if (first.isReady()) {
        if (1LL + jdim >= first.ndims() && first.isLinear()) {
            // first & out are linear
            getQueue().enqueueCopyBuffer(
                *first.get(), *out.get(), first.getOffset() * sizeof(T), 0,
                first.elements() * sizeof(T), nullptr, nullptr);
        } else {
            kernel::memcopy<T>(*out.get(), out.strides(), *first.get(), fdims,
                               first.strides(), first.getOffset(),
                               first.ndims(), 0);
        }
    } else {
        // Write the result directly in the out array
        const dim4 &outStrides{out.strides()};
        Param output{
            out.get(),
            {{fdims.dims[0], fdims.dims[1], fdims.dims[2], fdims.dims[3]},
             {outStrides.dims[0], outStrides.dims[1], outStrides.dims[2],
              outStrides.dims[3]},
             0}};
        evalNodes(output, first.getNode().get());
    }

    if (second.isReady()) {
        if (1LL + jdim >= second.ndims() && second.isLinear()) {
            // second & out are linear
            getQueue().enqueueCopyBuffer(
                *second.get(), *out.get(), second.getOffset() * sizeof(T),
                (fdims.dims[jdim] * out.strides().dims[jdim]) * sizeof(T),
                second.elements() * sizeof(T), nullptr, nullptr);
        } else {
            kernel::memcopy<T>(*out.get(), out.strides(), *second.get(), sdims,
                               second.strides(), second.getOffset(),
                               second.ndims(),
                               fdims.dims[jdim] * out.strides().dims[jdim]);
        }
    } else {
        // Write the result directly in the out array
        const dim4 &outStrides{out.strides()};
        Param output{
            out.get(),
            {{sdims.dims[0], sdims.dims[1], sdims.dims[2], sdims.dims[3]},
             {outStrides.dims[0], outStrides.dims[1], outStrides.dims[2],
              outStrides.dims[3]},
             fdims.dims[jdim] * outStrides.dims[jdim]}};
        evalNodes(output, second.getNode().get());
    }

    return out;
}

template<typename T>
void join(Array<T> &out, const int jdim, const vector<Array<T>> &inputs) {
    const dim_t n_arrays = inputs.size();
    if (n_arrays == 0) return;

    // avoid buffer overflow
    const dim4 &odims{out.dims()};
    const dim4 &fdims{inputs[0].dims()};
    // All dimensions of inputs needs to be equal except for the join
    // dimension
    assert(std::all_of(inputs.begin(), inputs.end(),
                       [jdim, &fdims](const Array<T> &in) {
                           bool eq{true};
                           for (int i = 0; i < 4; ++i) {
                               if (i != jdim) {
                                   eq &= fdims.dims[i] == in.dims().dims[i];
                               };
                           };
                           return eq;
                       }));
    // All dimensions of out needs to cover all input dimensions
    assert(
        (odims.dims[0] >= fdims.dims[0]) && (odims.dims[1] >= fdims.dims[1]) &&
        (odims.dims[2] >= fdims.dims[2]) && (odims.dims[3] >= fdims.dims[3]));
    // The join dimension of out needs to be larger than the
    // sum of all input join dimensions
    assert(odims.dims[jdim] >=
           std::accumulate(inputs.begin(), inputs.end(), 0,
                           [jdim](dim_t dim, const Array<T> &in) {
                               return dim += in.dims().dims[jdim];
                           }));
    assert(out.strides().dims[0] == 1);

    class eval {
       public:
        vector<Param> outputs;
        vector<Node_ptr> nodePtrs;
        vector<Node *> nodes;
        vector<const Array<T> *> ins;
    };
    std::map<dim_t, eval> evals;
    const dim4 &ostrides{out.strides()};
    const size_t L2CacheSize{getL2CacheSize(opencl::getDevice())};

    // topspeed is achieved when byte size(in+out) ~= L2CacheSize
    //
    // 1 array: memcpy always copies 1 array.  topspeed
    //      --> size(in) <= L2CacheSize/2
    // 2 arrays: topspeeds
    //      - size(in) < L2CacheSize/2/2
    //          --> JIT can copy 2 arrays in // and is fastest
    //              (condition: array sizes have to be identical)
    //      - size(in) < L2CacheSize/2
    //          --> memcpy will achieve highest speed, although the kernel
    //              has to be called twice
    //      - size(in) >= L2CacheSize/2
    //          --> memcpy will achieve veryLargeArray speed.  The kernel
    //              will be called twice

    // Group all arrays according to size
    dim_t odim{0}, outOffset{0};
    for (const Array<T> &iArray : inputs) {
        const dim4 &idims{iArray.dims()};
        for (int i = 0; i < AF_MAX_DIMS; ++i)
            ARG_ASSERT(1, odims.dims[i] >= idims.dims[i]);
        eval &e{evals[idims[jdim]]};
        const Param output{
            out.get(),
            {{idims.dims[0], idims.dims[1], idims.dims[2], idims.dims[3]},
             {ostrides.dims[0], ostrides.dims[1], ostrides.dims[2],
              ostrides.dims[3]},
             outOffset}};
        e.outputs.push_back(output);
        // Extend life of the returned node by saving the corresponding
        // shared_ptr
        e.nodePtrs.emplace_back(iArray.getNode());
        e.nodes.push_back(e.nodePtrs.back().get());
        e.ins.push_back(&iArray);
        odim += idims.dims[jdim];
        outOffset = odim * ostrides.dims[jdim];
    }

    for (auto &eval : evals) {
        auto &s{eval.second};
        if (s.ins.size() == 1 ||
            s.ins[0]->elements() * sizeof(T) * 2 * 2 > L2CacheSize) {
            // Process (evaluate arrays) individually for
            //  - single small array
            //  - very large arrays
            auto nodeIt{begin(s.nodes)};
            auto outputIt{begin(s.outputs)};
            for (const Array<T> *in : s.ins) {
                if (in->isReady()) {
                    const dim4 &istrides{in->strides()};
                    bool lin = in->isLinear() & (ostrides.dims[0] == 1);
                    for (int i{1}; i < in->ndims(); ++i) {
                        lin &= (ostrides.dims[i] == istrides.dims[i]);
                    }
                    if (lin) {
                        getQueue().enqueueCopyBuffer(
                            *in->get(), *outputIt->data,
                            in->getOffset() * sizeof(T),
                            outputIt->info.offset * sizeof(T),
                            in->elements() * sizeof(T), nullptr, nullptr);
                    } else {
                        kernel::memcopy<T>(*outputIt->data,
                                           af::dim4(4, outputIt->info.strides),
                                           *in->get(), in->dims(),
                                           in->strides(), in->getOffset(),
                                           in->ndims(), outputIt->info.offset);
                    }
                    // eliminate this array from the list, so that it will
                    // not be processed in bulk via JIT
                    outputIt = s.outputs.erase(outputIt);
                    nodeIt   = s.nodes.erase(nodeIt);
                } else {
                    ++outputIt;
                    ++nodeIt;
                }
            }
        }
        evalNodes(s.outputs, s.nodes);
    }
}

#define INSTANTIATE(T)                                               \
    template Array<T> join<T>(const int jdim, const Array<T> &first, \
                              const Array<T> &second);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(half)

#undef INSTANTIATE

#define INSTANTIATE(T)                                    \
    template void join<T>(Array<T> & out, const int jdim, \
                          const vector<Array<T>> &inputs);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(half)

#undef INSTANTIATE
}  // namespace opencl
}  // namespace arrayfire
