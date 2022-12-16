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
#include <index.hpp>
#include <sort.hpp>
#include <sort_index.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using arrayfire::common::half;
using std::iota;
using std::min;
using std::partial_sort_copy;
using std::vector;

namespace arrayfire {
namespace cpu {
template<typename T>
void topk(Array<T>& vals, Array<unsigned>& idxs, const Array<T>& in,
          const int k, const int dim, const af::topkFunction order) {
    // The out_dims is of size k along the dimension of the topk operation
    // and the same as the input dimension otherwise.
    dim4 out_dims(1);
    int ndims = in.dims().ndims();
    for (int i = 0; i < ndims; i++) {
        if (i == dim) {
            out_dims[i] = min(k, static_cast<int>(in.dims()[i]));
        } else {
            out_dims[i] = in.dims()[i];
        }
    }

    auto values  = createEmptyArray<T>(out_dims);
    auto indices = createEmptyArray<unsigned>(out_dims);

    auto func = [=](Param<T> values, Param<unsigned> indices, CParam<T> in) {
        const T* ptr   = in.get();
        unsigned* iptr = indices.get();
        T* vptr        = values.get();

        // Create a linear index
        vector<uint> idx(in.dims().elements());
        iota(begin(idx), end(idx), 0);

        int iter = in.dims()[1] * in.dims()[2] * in.dims()[3];
        for (int i = 0; i < iter; i++) {
            auto idx_itr = begin(idx) + i * in.strides()[1];
            auto* kiptr  = iptr + k * i;

            if (order == AF_TOPK_MIN) {
                // Sort the top k values in each column
                partial_sort_copy(
                    idx_itr, idx_itr + in.strides()[1], kiptr, kiptr + k,
                    [ptr](const uint lhs, const uint rhs) -> bool {
                        return compute_t<T>(ptr[lhs]) < compute_t<T>(ptr[rhs]);
                    });
            } else {
                partial_sort_copy(
                    idx_itr, idx_itr + in.strides()[1], kiptr, kiptr + k,
                    [ptr](const uint lhs, const uint rhs) -> bool {
                        return compute_t<T>(ptr[lhs]) >= compute_t<T>(ptr[rhs]);
                    });
            }

            auto* kvptr = vptr + k * i;
            for (int j = 0; j < k; j++) {
                // Update the value arrays with the original values
                kvptr[j] = ptr[kiptr[j]];
                // Convert linear indices back to column indices
                kiptr[j] -= i * in.strides()[1];
            }
        }
    };

    getQueue().enqueue(func, values, indices, in);

    vals = values;
    idxs = indices;
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
}  // namespace cpu
}  // namespace arrayfire
