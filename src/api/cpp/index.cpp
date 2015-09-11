/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/index.h>
#include <af/array.h>
#include <af/algorithm.h>
#include "error.hpp"
#include "common.hpp"

namespace af
{

array lookup(const array &in, const array &idx, const int dim)
{
    af_array out = 0;
    AF_THROW(af_lookup(&out, in.get(), idx.get(), getFNSD(dim, in.dims())));
    return array(out);
}

void copy(array &dst, const array &src,
          const index &idx0,
          const index &idx1,
          const index &idx2,
          const index &idx3)
{
    unsigned nd = dst.numdims();

    af_index_t indices[] = {idx0.get(),
                            idx1.get(),
                            idx2.get(),
                            idx3.get()};

    af_array lhs = dst.get();
    const af_array rhs = src.get();
    AF_THROW(af_assign_gen(&lhs, lhs, nd, indices, rhs));
}

index::index() {
    impl.idx.seq = af_span;
    impl.isSeq = true;
    impl.isBatch = false;
}

index::index(const int idx) {
    impl.idx.seq = af_make_seq(idx, idx, 1);
    impl.isSeq = true;
    impl.isBatch = false;
}

index::index(const af::seq& s0) {
    impl.idx.seq = s0.s;
    impl.isSeq = true;
    impl.isBatch = s0.m_gfor;
}

index::index(const af_seq& s0) {
    impl.idx.seq = s0;
    impl.isSeq = true;
    impl.isBatch = false;
}

index::index(const af::array& idx0) {
    array idx = idx0.isbool() ? where(idx0) : idx0;
    af_array arr = 0;
    AF_THROW(af_retain_array(&arr, idx.get()));
    impl.idx.arr = arr;

    impl.isSeq = false;
    impl.isBatch = false;
}

index::index(const af::index& idx0) {
    *this = idx0;
}

index::~index() {
    if (!impl.isSeq)
        af_release_array(impl.idx.arr);
}

index & index::operator=(const index& idx0) {
    impl = idx0.get();
    if(impl.isSeq == false){
        // increment reference count to avoid double free
        // when/if idx0 is destroyed
        AF_THROW(af_retain_array(&impl.idx.arr, impl.idx.arr));
    }
    return *this;
}

#if __cplusplus > 199711L
index::index(index &&idx0) {
    impl = idx0.impl;
}

index& index::operator=(index &&idx0) {
    impl = idx0.impl;
    return *this;
}
#endif


static bool operator==(const af_seq& lhs, const af_seq& rhs) {
    return lhs.begin == rhs.begin && lhs.end == rhs.end && lhs.step == rhs.step;
}

bool index::isspan() const
{
    return impl.isSeq == true && impl.idx.seq == af_span;
}

const af_index_t& index::get() const
{
    return impl;
}

}
