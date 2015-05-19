/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <sort.hpp>
#include <math.hpp>
#include <copy.hpp>
#include <stdexcept>
#include <err_cpu.hpp>
#include <algorithm>
#include <functional>

using std::greater;
using std::less;
using std::sort;
using std::function;

namespace cpu
{
    ///////////////////////////////////////////////////////////////////////////
    // Kernel Functions
    ///////////////////////////////////////////////////////////////////////////

    // Based off of http://stackoverflow.com/a/12399290
    template<typename T, bool isAscending>
    void sort0(Array<T> &val)
    {
        // initialize original index locations
        T *val_ptr = val.get();

        function<bool(T, T)> op = greater<T>();
        if(isAscending) { op = less<T>(); }

        T *comp_ptr = nullptr;
        for(dim_t w = 0; w < val.dims()[3]; w++) {
            dim_t valW = w * val.strides()[3];
            for(dim_t z = 0; z < val.dims()[2]; z++) {
                dim_t valWZ = valW + z * val.strides()[2];
                for(dim_t y = 0; y < val.dims()[1]; y++) {

                    dim_t valOffset = valWZ + y * val.strides()[1];

                    comp_ptr = val_ptr + valOffset;
                    std::sort(comp_ptr, comp_ptr + val.dims()[0], op);
                }
            }
        }
        return;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Wrapper Functions
    ///////////////////////////////////////////////////////////////////////////
    template<typename T, bool isAscending>
    Array<T> sort(const Array<T> &in, const unsigned dim)
    {
        Array<T> out = copyArray<T>(in);
        switch(dim) {
            case 0: sort0<T, isAscending>(out);
                    break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
        return out;
    }

#define INSTANTIATE(T)                                                  \
    template Array<T> sort<T, true>(const Array<T> &in, const unsigned dim); \
    template Array<T> sort<T,false>(const Array<T> &in, const unsigned dim); \

    INSTANTIATE(float)
    INSTANTIATE(double)
    //INSTANTIATE(cfloat)
    //INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
}
