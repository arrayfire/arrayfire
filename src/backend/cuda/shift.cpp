/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <JIT/ShiftNode.hpp>
#include <Array.hpp>
#include <shift.hpp>
#include <stdexcept>
#include <err_cuda.hpp>

namespace cuda
{
    template<typename T>
    Array<T> shift(const Array<T> &in, const int sdims[4])
    {

        // Shift should only be the first node in the JIT tree.
        // Force input to be evaluated so that in is always a buffer.
        in.eval();

        std::string name_str("Sh");
        name_str += shortname<T>(true);
        const af::dim4 iDims = in.dims();
        af::dim4 oDims = iDims;

        std::array<int, 4> shifts;
        for(int i = 0; i < 4; i++) {
            // sdims_[i] will always be positive and always [0, oDims[i]].
            // Negative shifts are converted to position by going the other way round
            shifts[i] = -(sdims[i] % (int)oDims[i]) + oDims[i] * (sdims[i] > 0);
            assert(shifts[i] >= 0 && shifts[i] <= oDims[i]);
        }

        auto node = new JIT::ShiftNode<T>(getFullName<T>(), name_str.c_str(),
                                          in.getNode(), shifts);
        return createNodeArray<T>(oDims, JIT::Node_ptr(node));
    }

#define INSTANTIATE(T)                                                  \
    template Array<T> shift<T>(const Array<T> &in, const int sdims[4]); \

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
}
