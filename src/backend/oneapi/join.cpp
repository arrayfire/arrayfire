/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <err_oneapi.hpp>
#include <join.hpp>
#include <kernel/memcopy.hpp>
#include <platform.hpp>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <vector>

using af::dim4;
using arrayfire::common::half;
using arrayfire::common::Node;
using arrayfire::common::Node_ptr;
using std::transform;
using std::vector;

namespace arrayfire {
namespace oneapi {
dim4 calcOffset(const dim4 &dims, int dim) {
    dim4 offset;
    offset[0] = (dim == 0) ? dims[0] : 0;
    offset[1] = (dim == 1) ? dims[1] : 0;
    offset[2] = (dim == 2) ? dims[2] : 0;
    offset[3] = (dim == 3) ? dims[3] : 0;
    return offset;
}

template<typename T>
Array<T> join(const int jdim, const Array<T> &first, const Array<T> &second) {
    // All dimensions except join dimension must be equal
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
        const size_t L2CacheSize{getL2CacheSize(oneapi::getDevice())};
        if (!(first.isReady() || second.isReady()) ||
            (fdims.elements() * sizeof(T) * 2 * 2 < L2CacheSize)) {
            // Both arrays have same size & everything fits into the cache,
            // so thread in 1 JIT kernel, iso individual copies which is
            // always slower
            const dim4 &outStrides{out.strides()};
            vector<Param<T>> outputs{
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
            getQueue().submit([&](sycl::handler &h) {
                sycl::range sz(first.elements());
                sycl::id src_offset(first.getOffset());
                sycl::accessor offset_acc_src =
                    first.get()->template get_access<sycl::access_mode::read>(
                        h, sz, src_offset);
                sycl::id dst_offset(0);
                sycl::accessor offset_acc_dst =
                    out.get()->template get_access<sycl::access_mode::write>(
                        h, sz, dst_offset);
                h.copy(offset_acc_src, offset_acc_dst);
            });
        } else {
            kernel::memcopy<T>(out.get(), out.strides().get(), first.get(),
                               fdims.get(), first.strides().get(),
                               first.getOffset(), first.ndims());
        }
    } else {
        // Write the result directly in the out array
        const dim4 &outStrides{out.strides()};
        Param<T> output{
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
            getQueue().submit([&](sycl::handler &h) {
                sycl::range sz(second.elements());
                sycl::id src_offset(second.getOffset());
                sycl::accessor offset_acc_src =
                    second.get()->template get_access<sycl::access_mode::read>(
                        h, sz, src_offset);
                sycl::id dst_offset(fdims.dims[jdim] *
                                    out.strides().dims[jdim]);
                sycl::accessor offset_acc_dst =
                    out.get()->template get_access<sycl::access_mode::write>(
                        h, sz, dst_offset);
                h.copy(offset_acc_src, offset_acc_dst);
            });
        } else {
            kernel::memcopy<T>(out.get(), out.strides().get(), second.get(),
                               sdims.get(), second.strides().get(),
                               second.getOffset(), second.ndims(),
                               fdims.dims[jdim] * out.strides().dims[jdim]);
        }
    } else {
        // Write the result directly in the out array
        const dim4 &outStrides{out.strides()};
        Param<T> output{
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
        vector<Param<T>> outputs;
        vector<Node_ptr> nodePtrs;
        vector<Node *> nodes;
        vector<const Array<T> *> ins;
    };
    std::map<dim_t, eval> evals;
    const dim4 &ostrides{out.strides()};
    const size_t L2CacheSize{getL2CacheSize(oneapi::getDevice())};

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
    dim_t outOffset{0};
    for (const Array<T> &iArray : inputs) {
        const dim4 &idims{iArray.dims()};
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
        outOffset += idims.dims[jdim] * ostrides.dims[jdim];
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
                    const dim_t *istrides{in->strides().dims};
                    bool lin = in->isLinear() & (ostrides.dims[0] == 1);
                    for (int i{1}; i < in->ndims(); ++i) {
                        lin &= (ostrides.dims[i] == istrides[i]);
                    }
                    if (lin) {
                        getQueue().submit([&](sycl::handler &h) {
                            sycl::range sz(in->elements());
                            sycl::id src_offset(in->getOffset());
                            sycl::accessor offset_acc_src =
                                in->get()
                                    ->template get_access<
                                        sycl::access_mode::read>(h, sz,
                                                                 src_offset);
                            sycl::id dst_offset(outputIt->info.offset);
                            sycl::accessor offset_acc_dst =
                                outputIt->data->template get_access<
                                    sycl::access_mode::write>(h, sz,
                                                              dst_offset);
                            h.copy(offset_acc_src, offset_acc_dst);
                        });
                    } else {
                        kernel::memcopy<T>(
                            outputIt->data,
                            af::dim4(4, outputIt->info.strides).get(),
                            in->get(), in->dims().get(), in->strides().get(),
                            in->getOffset(), in->ndims(),
                            outputIt->info.offset);
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

#define INSTANTIATE(T)                                              \
    template Array<T> join<T>(const int dim, const Array<T> &first, \
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

#define INSTANTIATE(T)                                   \
    template void join<T>(Array<T> & out, const int dim, \
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
}  // namespace oneapi
}  // namespace arrayfire
