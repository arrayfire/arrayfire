/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <shift.hpp>

#include <common/jit/ShiftNodeBase.hpp>
#include <err_oneapi.hpp>
#include <traits.hpp>

using af::dim4;
using arrayfire::common::Node_ptr;
using arrayfire::common::ShiftNodeBase;
using std::array;
using std::make_shared;
using std::static_pointer_cast;
using std::string;

namespace arrayfire {
namespace oneapi {
template<typename T>
using ShiftNode = ShiftNodeBase<jit::BufferNode<T>>;

template<typename T>
Array<T> shift(const Array<T> &in, const int sdims[4]) {
    // Shift should only be the first node in the JIT tree.
    // Force input to be evaluated so that in is always a buffer.
    in.eval();

    string name_str("Sh");
    name_str += shortname<T>(true);
    const dim4 &iDims = in.dims();
    dim4 oDims        = iDims;

    array<int, 4> shifts{};
    for (int i = 0; i < 4; i++) {
        // sdims_[i] will always be positive and always [0, oDims[i]].
        // Negative shifts are converted to position by going the other way
        // round
        shifts[i] = -(sdims[i] % static_cast<int>(oDims[i])) +
                    oDims[i] * (sdims[i] > 0);
        assert(shifts[i] >= 0 && shifts[i] <= oDims[i]);
    }

    auto node = make_shared<ShiftNode<T>>(
        static_cast<af::dtype>(dtype_traits<T>::af_type),
        static_pointer_cast<jit::BufferNode<T>>(in.getNode()), shifts);
    return createNodeArray<T>(oDims, common::Node_ptr(node));
}

#define INSTANTIATE(T) \
    template Array<T> shift<T>(const Array<T> &in, const int sdims[4]);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(short)
INSTANTIATE(ushort)
}  // namespace oneapi
}  // namespace arrayfire
