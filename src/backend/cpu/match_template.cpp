/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <kernel/match_template.hpp>
#include <match_template.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace cpu {

template<typename InT, typename OutT, af_match_type MatchT>
Array<OutT> match_template(const Array<InT> &sImg, const Array<InT> &tImg) {
    Array<OutT> out = createEmptyArray<OutT>(sImg.dims());

    getQueue().enqueue(kernel::matchTemplate<OutT, InT, MatchT>, out, sImg,
                       tImg);

    return out;
}

#define INSTANTIATE(in_t, out_t)                                \
    template Array<out_t> match_template<in_t, out_t, AF_SAD>(  \
        const Array<in_t> &sImg, const Array<in_t> &tImg);      \
    template Array<out_t> match_template<in_t, out_t, AF_LSAD>( \
        const Array<in_t> &sImg, const Array<in_t> &tImg);      \
    template Array<out_t> match_template<in_t, out_t, AF_ZSAD>( \
        const Array<in_t> &sImg, const Array<in_t> &tImg);      \
    template Array<out_t> match_template<in_t, out_t, AF_SSD>(  \
        const Array<in_t> &sImg, const Array<in_t> &tImg);      \
    template Array<out_t> match_template<in_t, out_t, AF_LSSD>( \
        const Array<in_t> &sImg, const Array<in_t> &tImg);      \
    template Array<out_t> match_template<in_t, out_t, AF_ZSSD>( \
        const Array<in_t> &sImg, const Array<in_t> &tImg);      \
    template Array<out_t> match_template<in_t, out_t, AF_NCC>(  \
        const Array<in_t> &sImg, const Array<in_t> &tImg);      \
    template Array<out_t> match_template<in_t, out_t, AF_ZNCC>( \
        const Array<in_t> &sImg, const Array<in_t> &tImg);      \
    template Array<out_t> match_template<in_t, out_t, AF_SHD>(  \
        const Array<in_t> &sImg, const Array<in_t> &tImg);

INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(char, float)
INSTANTIATE(int, float)
INSTANTIATE(uint, float)
INSTANTIATE(uchar, float)
INSTANTIATE(short, float)
INSTANTIATE(ushort, float)

}  // namespace cpu
