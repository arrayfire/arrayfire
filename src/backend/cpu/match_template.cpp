/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <match_template.hpp>

#include <kernel/match_template.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

#include <functional>

using af::dim4;

namespace arrayfire {
namespace cpu {

template<typename To, typename Ti>
using matchFunc = std::function<void(Param<To>, CParam<Ti>, CParam<Ti>)>;

template<typename inType, typename outType>
Array<outType> match_template(const Array<inType> &sImg,
                              const Array<inType> &tImg,
                              const af::matchType mType) {
    static const matchFunc<outType, inType> funcs[6] = {
        kernel::matchTemplate<outType, inType, AF_SAD>,
        kernel::matchTemplate<outType, inType, AF_ZSAD>,
        kernel::matchTemplate<outType, inType, AF_LSAD>,
        kernel::matchTemplate<outType, inType, AF_SSD>,
        kernel::matchTemplate<outType, inType, AF_ZSSD>,
        kernel::matchTemplate<outType, inType, AF_LSSD>,
    };

    Array<outType> out = createEmptyArray<outType>(sImg.dims());
    getQueue().enqueue(funcs[static_cast<int>(mType)], out, sImg, tImg);
    return out;
}

#define INSTANTIATE(in_t, out_t)                       \
    template Array<out_t> match_template<in_t, out_t>( \
        const Array<in_t> &, const Array<in_t> &, const af::matchType);

INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(char, float)
INSTANTIATE(int, float)
INSTANTIATE(uint, float)
INSTANTIATE(uchar, float)
INSTANTIATE(short, float)
INSTANTIATE(ushort, float)

}  // namespace cpu
}  // namespace arrayfire
