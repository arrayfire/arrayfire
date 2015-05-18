/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <sort_by_key.hpp>
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

    template<typename Tk, typename Tv, bool isAscending>
    void sort0_by_key(Array<Tk> &okey, Array<Tv> &oval, const Array<Tk> &ikey, const Array<Tv> &ival)
    {
        function<bool(Tk, Tk)> op = greater<Tk>();
        if(isAscending) { op = less<Tk>(); }

        // Get pointers and initialize original index locations
        Array<uint> oidx = createValueArray(ikey.dims(), 0u);
            uint *oidx_ptr = oidx.get();
              Tk *okey_ptr = okey.get();
              Tv *oval_ptr = oval.get();
        const Tk *ikey_ptr = ikey.get();
        const Tv *ival_ptr = ival.get();

        std::vector<uint> seq_vec(oidx.dims()[0]);
        std::iota(seq_vec.begin(), seq_vec.end(), 0);

        const Tk *comp_ptr = nullptr;
        auto comparator = [&comp_ptr, &op](size_t i1, size_t i2) {return op(comp_ptr[i1], comp_ptr[i2]);};

        for(dim_t w = 0; w < ikey.dims()[3]; w++) {
            dim_t okeyW = w * okey.strides()[3];
            dim_t ovalW = w * oval.strides()[3];
            dim_t oidxW = w * oidx.strides()[3];
            dim_t ikeyW = w * ikey.strides()[3];
            dim_t ivalW = w * ival.strides()[3];

            for(dim_t z = 0; z < ikey.dims()[2]; z++) {
                dim_t okeyWZ = okeyW + z * okey.strides()[2];
                dim_t ovalWZ = ovalW + z * oval.strides()[2];
                dim_t oidxWZ = oidxW + z * oidx.strides()[2];
                dim_t ikeyWZ = ikeyW + z * ikey.strides()[2];
                dim_t ivalWZ = ivalW + z * ival.strides()[2];

                for(dim_t y = 0; y < ikey.dims()[1]; y++) {

                    dim_t okeyOffset = okeyWZ + y * okey.strides()[1];
                    dim_t ovalOffset = ovalWZ + y * oval.strides()[1];
                    dim_t oidxOffset = oidxWZ + y * oidx.strides()[1];
                    dim_t ikeyOffset = ikeyWZ + y * ikey.strides()[1];
                    dim_t ivalOffset = ivalWZ + y * ival.strides()[1];

                    uint *ptr = oidx_ptr + oidxOffset;
                    std::copy(seq_vec.begin(), seq_vec.end(), ptr);

                    comp_ptr = ikey_ptr + ikeyOffset;
                    std::stable_sort(ptr, ptr + ikey.dims()[0], comparator);

                    for (dim_t i = 0; i < oval.dims()[0]; ++i){
                        uint sortIdx = oidx_ptr[oidxOffset + i];
                        okey_ptr[okeyOffset + i] = ikey_ptr[ikeyOffset + sortIdx];
                        oval_ptr[ovalOffset + i] = ival_ptr[ivalOffset + sortIdx];
                    }
                }
            }
        }

        return;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Wrapper Functions
    ///////////////////////////////////////////////////////////////////////////
    template<typename Tk, typename Tv, bool isAscending>
    void sort_by_key(Array<Tk> &okey, Array<Tv> &oval,
               const Array<Tk> &ikey, const Array<Tv> &ival, const uint dim)
    {
        okey = createEmptyArray<Tk>(ikey.dims());
        oval = createEmptyArray<Tv>(ival.dims());
        switch(dim) {
            case 0: sort0_by_key<Tk, Tv, isAscending>(okey, oval, ikey, ival);
                    break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

#define INSTANTIATE(Tk, Tv)                                             \
    template void                                                       \
    sort_by_key<Tk, Tv, true>(Array<Tk> &okey, Array<Tv> &oval,         \
                              const Array<Tk> &ikey, const Array<Tv> &ival, const uint dim); \
    template void                                                       \
    sort_by_key<Tk, Tv,false>(Array<Tk> &okey, Array<Tv> &oval,         \
                              const Array<Tk> &ikey, const Array<Tv> &ival, const uint dim); \

#define INSTANTIATE1(Tk)       \
    INSTANTIATE(Tk, float)     \
    INSTANTIATE(Tk, double)    \
    INSTANTIATE(Tk, int)       \
    INSTANTIATE(Tk, uint)      \
    INSTANTIATE(Tk, char)      \
    INSTANTIATE(Tk, uchar)     \

    INSTANTIATE1(float)
    INSTANTIATE1(double)
    INSTANTIATE1(int)
    INSTANTIATE1(uint)
    INSTANTIATE1(char)
    INSTANTIATE1(uchar)
}
