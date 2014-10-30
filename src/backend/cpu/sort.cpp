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
#include <stdexcept>
#include <err_cpu.hpp>
#include <algorithm>
#include <queue>
#include <future>

using std::greater;
using std::less;
using std::sort;
using std::function;
using std::queue;
using std::future;
using std::async;

namespace cpu
{
    ///////////////////////////////////////////////////////////////////////////
    // Kernel Functions
    ///////////////////////////////////////////////////////////////////////////

    // Based off of http://stackoverflow.com/a/12399290
    template<typename T, bool DIR>
    void sort0(Array<T> &val, const Array<T> &in)
    {
        // initialize original index locations
              T *val_ptr = val.get();

        function<bool(dim_type, dim_type)> op = greater<T>();
        if(DIR) { op = less<T>(); }

        T *comp_ptr = nullptr;
        for(dim_type w = 0; w < in.dims()[3]; w++) {
            dim_type valW = w * val.strides()[3];
            for(dim_type z = 0; z < in.dims()[2]; z++) {
                dim_type valWZ = valW + z * val.strides()[2];
                for(dim_type y = 0; y < in.dims()[1]; y++) {

                    dim_type valOffset = valWZ + y * val.strides()[1];

                    comp_ptr = val_ptr + valOffset;
                    std::stable_sort(comp_ptr, comp_ptr + val.dims()[0], op);
                }
            }
        }
        return;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Wrapper Functions
    ///////////////////////////////////////////////////////////////////////////
    template<typename T, bool DIR>
    void sort(Array<T> &val, const Array<T> &in, const unsigned dim)
    {
        switch(dim) {
            case 0: sort0<T, DIR>(val, in);
                    break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

#define INSTANTIATE(T)                                                                          \
    template void sort<T, true>(Array<T> &val, const Array<T> &in, const unsigned dim);         \
    template void sort<T,false>(Array<T> &val, const Array<T> &in, const unsigned dim);         \

    INSTANTIATE(float)
    INSTANTIATE(double)
    //INSTANTIATE(cfloat)
    //INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
}
