/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <sort_index.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_cpu.hpp>
#include <algorithm>
#include <numeric>
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
    template<typename T, bool isAscending>
    void sort0_index(Array<T> &val, Array<uint> &idx, const Array<T> &in)
    {
        // initialize original index locations
           uint *idx_ptr = idx.get();
              T *val_ptr = val.get();
        const T *in_ptr  = in.get();
        function<bool(T, T)> op = greater<T>();
        if(isAscending) { op = less<T>(); }

        std::vector<uint> seq_vec(idx.dims()[0]);
        std::iota(seq_vec.begin(), seq_vec.end(), 0);

        const T *comp_ptr = nullptr;
        auto comparator = [&comp_ptr, &op](size_t i1, size_t i2) {return op(comp_ptr[i1], comp_ptr[i2]);};

        for(dim_t w = 0; w < in.dims()[3]; w++) {
            dim_t valW = w * val.strides()[3];
            dim_t idxW = w * idx.strides()[3];
            dim_t  inW = w *  in.strides()[3];
            for(dim_t z = 0; z < in.dims()[2]; z++) {
                dim_t valWZ = valW + z * val.strides()[2];
                dim_t idxWZ = idxW + z * idx.strides()[2];
                dim_t  inWZ =  inW + z *  in.strides()[2];
                for(dim_t y = 0; y < in.dims()[1]; y++) {

                    dim_t valOffset = valWZ + y * val.strides()[1];
                    dim_t idxOffset = idxWZ + y * idx.strides()[1];
                    dim_t inOffset  =  inWZ + y *  in.strides()[1];

                    uint *ptr = idx_ptr + idxOffset;
                    std::copy(seq_vec.begin(), seq_vec.end(), ptr);

                    comp_ptr = in_ptr + inOffset;
                    std::stable_sort(ptr, ptr + in.dims()[0], comparator);

                    for (dim_t i = 0; i < val.dims()[0]; ++i){
                        val_ptr[valOffset + i] = in_ptr[inOffset + idx_ptr[idxOffset + i]];
                    }
                }
            }
        }

        return;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Wrapper Functions
    ///////////////////////////////////////////////////////////////////////////
    template<typename T, bool isAscending>
    void sort_index(Array<T> &val, Array<uint> &idx, const Array<T> &in, const uint dim)
    {
        val = createEmptyArray<T>(in.dims());
        idx = createEmptyArray<uint>(in.dims());
        switch(dim) {
            case 0: sort0_index<T, isAscending>(val, idx, in);
                    break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

#define INSTANTIATE(T)                                                  \
    template void sort_index<T, true>(Array<T> &val, Array<uint> &idx, const Array<T> &in, \
                                      const uint dim);                  \
    template void sort_index<T,false>(Array<T> &val, Array<uint> &idx, const Array<T> &in, \
                                      const uint dim);                  \

    INSTANTIATE(float)
    INSTANTIATE(double)
    //INSTANTIATE(cfloat)
    //INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
}
