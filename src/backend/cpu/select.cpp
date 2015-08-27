/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <select.hpp>
#include <err_cpu.hpp>

using af::dim4;

namespace cpu
{
    template<typename T>
    void select(Array<T> &out, const Array<char> &cond, const Array<T> &a, const Array<T> &b)
    {
        dim4 adims = a.dims();
        dim4 astrides = a.strides();
        dim4 bdims = b.dims();
        dim4 bstrides = b.strides();

        dim4 cdims = cond.dims();
        dim4 cstrides = cond.strides();

        dim4 odims = out.dims();
        dim4 ostrides = out.strides();

        bool is_a_same[] = {adims[0] == odims[0], adims[1] == odims[1],
                            adims[2] == odims[2], adims[3] == odims[3]};

        bool is_b_same[] = {bdims[0] == odims[0], bdims[1] == odims[1],
                            bdims[2] == odims[2], bdims[3] == odims[3]};

        bool is_c_same[] = {cdims[0] == odims[0], cdims[1] == odims[1],
                            cdims[2] == odims[2], cdims[3] == odims[3]};

        const T *aptr = a.get();
        const T *bptr = b.get();
        T *optr = out.get();
        const char *cptr = cond.get();

        for (int l = 0; l < odims[3]; l++) {

            int o_off3   = ostrides[3] * l;
            int a_off3   = astrides[3] * is_a_same[3] * l;
            int b_off3   = bstrides[3] * is_b_same[3] * l;
            int c_off3   = cstrides[3] * is_c_same[3] * l;

            for (int k = 0; k < odims[2]; k++) {

                int o_off2   = ostrides[2] * k + o_off3;
                int a_off2   = astrides[2] * is_a_same[2] * k + a_off3;
                int b_off2   = bstrides[2] * is_b_same[2] * k + b_off3;
                int c_off2   = cstrides[2] * is_c_same[2] * k + c_off3;

                for (int j = 0; j < odims[1]; j++) {

                    int o_off1   = ostrides[1] * j + o_off2;
                    int a_off1   = astrides[1] * is_a_same[1] * j + a_off2;
                    int b_off1   = bstrides[1] * is_b_same[1] * j + b_off2;
                    int c_off1   = cstrides[1] * is_c_same[1] * j + c_off2;

                    for (int i = 0; i < odims[0]; i++) {

                        bool cval = is_c_same[0] ? cptr[c_off1 + i] : cptr[c_off1];
                        T    aval = is_a_same[0] ? aptr[a_off1 + i] : aptr[a_off1];
                        T    bval = is_b_same[0] ? bptr[b_off1 + i] : bptr[b_off1];
                        T    oval = cval ? aval : bval;
                        optr[o_off1 + i] = oval;
                    }
                }
            }
        }
    }

    template<typename T, bool flip>
    void select_scalar(Array<T> &out, const Array<char> &cond, const Array<T> &a, const double &b)
    {
        dim4 astrides = a.strides();
        dim4 cstrides = cond.strides();

        dim4 odims = out.dims();
        dim4 ostrides = out.strides();

        const T *aptr = a.get();
        T *optr = out.get();
        const char *cptr = cond.get();

        for (int l = 0; l < odims[3]; l++) {

            int o_off3 = ostrides[3] * l;
            int a_off3 = astrides[3] * l;
            int c_off3 = cstrides[3] * l;

            for (int k = 0; k < odims[2]; k++) {

                int o_off2 = ostrides[2] * k + o_off3;
                int a_off2 = astrides[2] * k + a_off3;
                int c_off2 = cstrides[2] * k + c_off3;

                for (int j = 0; j < odims[1]; j++) {

                    int o_off1 = ostrides[1] * j + o_off2;
                    int a_off1 = astrides[1] * j + a_off2;
                    int c_off1 = cstrides[1] * j + c_off2;

                    for (int i = 0; i < odims[0]; i++) {

                        optr[o_off1 + i] = (flip ^ cptr[c_off1 + i]) ? aptr[a_off1 + i] : b;
                    }
                }
            }
        }
    }


#define INSTANTIATE(T)                                              \
    template void select<T>(Array<T> &out, const Array<char> &cond, \
                            const Array<T> &a, const Array<T> &b);  \
    template void select_scalar<T, true >(Array<T> &out,            \
                                          const Array<char> &cond,  \
                                          const Array<T> &a,        \
                                          const double &b);         \
    template void select_scalar<T, false>(Array<T> &out, const      \
                                          Array<char> &cond,        \
                                          const Array<T> &a,        \
                                          const double &b);         \

    INSTANTIATE(float  )
    INSTANTIATE(double )
    INSTANTIATE(cfloat )
    INSTANTIATE(cdouble)
    INSTANTIATE(int    )
    INSTANTIATE(uint   )
    INSTANTIATE(intl   )
    INSTANTIATE(uintl  )
    INSTANTIATE(char   )
    INSTANTIATE(uchar  )
}
