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
#include <index.hpp>
#include <sort.hpp>
#include <sort_index.hpp>
#include <types.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using arrayfire::common::half;

using std::iota;
using std::min;
using std::partial_sort_copy;
using std::transform;
using std::vector;

namespace arrayfire {
namespace oneapi {
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
    auto values  = createEmptyArray<T>(in.dims());
    auto indices = createEmptyArray<unsigned>(in.dims());
    sort_index(values, indices, in, dim, order & AF_TOPK_MIN);
    auto indVec = indexForTopK(k);
    vals        = index<T>(values, indVec.data());
    idxs        = index<unsigned>(indices, indVec.data());
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

}  // namespace oneapi
}  // namespace arrayfire
