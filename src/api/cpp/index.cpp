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

index::~index() {
    if (!impl.isSeq)
        af_release_array(impl.idx.arr);
}


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
