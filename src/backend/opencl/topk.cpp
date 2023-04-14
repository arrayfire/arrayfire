/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <err_opencl.hpp>
#include <index.hpp>
#include <sort.hpp>
#include <sort_index.hpp>
#include <types.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using arrayfire::common::half;
using cl::Buffer;
using cl::Event;

using std::iota;
using std::min;
using std::partial_sort_copy;
using std::transform;
using std::vector;

namespace arrayfire {
namespace opencl {
vector<af_index_t> indexForTopK(const int k) {
    af_index_t idx;
    idx.idx.seq = af_seq{0.0, static_cast<double>(k) - 1.0, 1.0};
    idx.isSeq   = true;
    idx.isBatch = false;

    af_index_t sp;
    sp.idx.seq = af_span;
    sp.isSeq   = true;
    sp.isBatch = false;

    return vector<af_index_t>({idx, sp, sp, sp});
}

template<typename T>
void topk(Array<T>& vals, Array<unsigned>& idxs, const Array<T>& in,
          const int k, const int dim, const af::topkFunction order) {
    if (getDeviceType() == CL_DEVICE_TYPE_CPU) {
        // This branch optimizes for CPU devices by first mapping the buffer
        // and calling partial sort on the buffer

        // TODO(umar): implement this in the kernel namespace

        // The out_dims is of size k along the dimension of the topk operation
        // and the same as the input dimension otherwise.
        dim4 out_dims(1);
        int ndims = in.dims().ndims();
        for (int i = 0; i < ndims; i++) {
            if (i == dim) {
                out_dims[i] = min(k, (int)in.dims()[i]);
            } else {
                out_dims[i] = in.dims()[i];
            }
        }

        auto values          = createEmptyArray<T>(out_dims);
        auto indices         = createEmptyArray<unsigned>(out_dims);
        const Buffer* in_buf = in.get();
        Buffer* ibuf         = indices.get();
        Buffer* vbuf         = values.get();

        cl::Event ev_in, ev_val, ev_ind;

        T* ptr     = static_cast<T*>(getQueue().enqueueMapBuffer(
            *in_buf, CL_FALSE, CL_MAP_READ, 0, in.elements() * sizeof(T),
            nullptr, &ev_in));
        uint* iptr = static_cast<uint*>(getQueue().enqueueMapBuffer(
            *ibuf, CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, k * sizeof(uint),
            nullptr, &ev_ind));
        T* vptr    = static_cast<T*>(getQueue().enqueueMapBuffer(
            *vbuf, CL_FALSE, CL_MAP_WRITE, 0, k * sizeof(T), nullptr, &ev_val));

        vector<uint> idx(in.elements());

        // Create a linear index
        iota(begin(idx), end(idx), 0);
        cl::Event::waitForEvents({ev_in, ev_ind});

        int iter = in.dims()[1] * in.dims()[2] * in.dims()[3];
        for (int i = 0; i < iter; i++) {
            auto idx_itr = begin(idx) + i * in.strides()[1];
            auto kiptr   = iptr + k * i;

            if (order & AF_TOPK_MIN) {
                if (order & AF_TOPK_STABLE) {
                    partial_sort_copy(
                        idx_itr, idx_itr + in.strides()[1], kiptr, kiptr + k,
                        [ptr](const uint lhs, const uint rhs) -> bool {
                            return (compute_t<T>(ptr[lhs]) <
                                    compute_t<T>(ptr[rhs]))
                                       ? true
                                   : compute_t<T>(ptr[lhs]) ==
                                           compute_t<T>(ptr[rhs])
                                       ? (lhs < rhs)
                                       : false;
                        });
                } else {
                    // Sort the top k values in each column
                    partial_sort_copy(
                        idx_itr, idx_itr + in.strides()[1], kiptr, kiptr + k,
                        [ptr](const uint lhs, const uint rhs) -> bool {
                            return compute_t<T>(ptr[lhs]) <
                                   compute_t<T>(ptr[rhs]);
                        });
                }
            } else {
                if (order & AF_TOPK_STABLE) {
                    partial_sort_copy(
                        idx_itr, idx_itr + in.strides()[1], kiptr, kiptr + k,
                        [ptr](const uint lhs, const uint rhs) -> bool {
                            return (compute_t<T>(ptr[lhs]) >
                                    compute_t<T>(ptr[rhs]))
                                       ? true
                                   : compute_t<T>(ptr[lhs]) ==
                                           compute_t<T>(ptr[rhs])
                                       ? (lhs < rhs)
                                       : false;
                        });
                } else {
                    partial_sort_copy(
                        idx_itr, idx_itr + in.strides()[1], kiptr, kiptr + k,
                        [ptr](const uint lhs, const uint rhs) -> bool {
                            return compute_t<T>(ptr[lhs]) >
                                   compute_t<T>(ptr[rhs]);
                        });
                }
            }
            ev_val.wait();

            auto kvptr = vptr + k * i;
            for (int j = 0; j < k; j++) {
                // Update the value arrays with the original values
                kvptr[j] = ptr[kiptr[j]];
                // Convert linear indices back to column indices
                kiptr[j] -= i * in.strides()[1];
            }
        }

        getQueue().enqueueUnmapMemObject(*ibuf, iptr);
        getQueue().enqueueUnmapMemObject(*vbuf, vptr);
        getQueue().enqueueUnmapMemObject(*in_buf, ptr);

        vals = values;
        idxs = indices;
    } else {
        auto values  = createEmptyArray<T>(in.dims());
        auto indices = createEmptyArray<unsigned>(in.dims());
        sort_index(values, indices, in, dim, order & AF_TOPK_MIN);
        auto indVec = indexForTopK(k);
        vals        = index<T>(values, indVec.data());
        idxs        = index<unsigned>(indices, indVec.data());
    }
}

#define INSTANTIATE(T)                                                  \
    template void topk<T>(Array<T>&, Array<unsigned>&, const Array<T>&, \
                          const int, const int, const af::topkFunction);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(long long)
INSTANTIATE(unsigned long long)
INSTANTIATE(half)
}  // namespace opencl
}  // namespace arrayfire
