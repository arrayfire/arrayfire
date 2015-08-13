/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <complex>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <memory.hpp>
#include <where.hpp>
#include <ops.hpp>
#include <vector>

using af::dim4;

namespace cpu
{
    template<typename T>
    Array<uint> where(const Array<T> &in)
    {
        const dim_t *dims    = in.dims().get();
        const dim_t *strides = in.strides().get();
        static const T zero = scalar<T>(0);

        const T *iptr = in.get();
        uint *out_vec  = memAlloc<uint>(in.elements());

        dim_t count = 0;
        dim_t idx = 0;
        for (dim_t w = 0; w < dims[3]; w++) {
            uint offw = w * strides[3];

            for (dim_t z = 0; z < dims[2]; z++) {
                uint offz = offw + z * strides[2];

                for (dim_t y = 0; y < dims[1]; y++) {
                    uint offy = y * strides[1] + offz;

                    for (dim_t x = 0; x < dims[0]; x++) {

                        T val = iptr[offy + x];
                        if (val != zero) {
                            out_vec[count] = idx;
                            count++;
                        }
                        idx++;
                    }
                }
            }
        }

        Array<uint> out = createDeviceDataArray<uint>(dim4(count), out_vec);
        return out;
    }

#define INSTANTIATE(T)                                  \
    template Array<uint> where<T>(const Array<T> &in);    \

    INSTANTIATE(float  )
    INSTANTIATE(cfloat )
    INSTANTIATE(double )
    INSTANTIATE(cdouble)
    INSTANTIATE(char   )
    INSTANTIATE(int    )
    INSTANTIATE(uint   )
    INSTANTIATE(intl   )
    INSTANTIATE(uintl  )
    INSTANTIATE(uchar  )

}
