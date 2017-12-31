/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// This file instantiates scan.cu as separate object files from CMake
// The line below is read by CMake to determenine the instantiations
// SCAN_BINARY_OPS:af_add_t af_mul_t af_max_t af_min_t af_notzero_t

#include <af/dim4.hpp>
#include <Array.hpp>
#include <err_cuda.hpp>

#undef _GLIBCXX_USE_INT128
#include <scan.hpp>
#include <complex.hpp>
#include <kernel/scan_first.hpp>
#include <kernel/scan_dim.hpp>

namespace cuda
{
    template<af_op_t op, typename Ti, typename To>
    Array<To> scan(const Array<Ti>& in, const int dim, bool inclusive_scan)
    {
        Array<To> out = createEmptyArray<To>(in.dims());

        if (inclusive_scan) {
            switch (dim) {
            case 0: kernel::scan_first<Ti, To, op   , true>(out, in); break;
            case 1: kernel::scan_dim  <Ti, To, op, 1, true>(out, in); break;
            case 2: kernel::scan_dim  <Ti, To, op, 2, true>(out, in); break;
            case 3: kernel::scan_dim  <Ti, To, op, 3, true>(out, in); break;
            }
        } else {
            switch (dim) {
            case 0: kernel::scan_first<Ti, To, op   , false>(out, in); break;
            case 1: kernel::scan_dim  <Ti, To, op, 1, false>(out, in); break;
            case 2: kernel::scan_dim  <Ti, To, op, 2, false>(out, in); break;
            case 3: kernel::scan_dim  <Ti, To, op, 3, false>(out, in); break;
            }
        }

        return out;
    }

#define INSTANTIATE_SCAN(ROp, Ti, To)\
    template Array<To> scan<ROp, Ti, To>(const Array<Ti> &in, const int dim, bool inclusive_scan);

#define INSTANTIATE_SCAN_ALL(ROp)                       \
    INSTANTIATE_SCAN(ROp, float  , float  )             \
    INSTANTIATE_SCAN(ROp, double , double )             \
    INSTANTIATE_SCAN(ROp, cfloat , cfloat )             \
    INSTANTIATE_SCAN(ROp, cdouble, cdouble)             \
    INSTANTIATE_SCAN(ROp, int    , int    )             \
    INSTANTIATE_SCAN(ROp, uint   , uint   )             \
    INSTANTIATE_SCAN(ROp, intl   , intl   )             \
    INSTANTIATE_SCAN(ROp, uintl  , uintl  )             \
    INSTANTIATE_SCAN(ROp, char   , int    )             \
    INSTANTIATE_SCAN(ROp, char   , uint   )             \
    INSTANTIATE_SCAN(ROp, uchar  , uint   )             \
    INSTANTIATE_SCAN(ROp, short  , int    )             \
    INSTANTIATE_SCAN(ROp, ushort , uint   )

#if SCAN_BINARY_OP == af_notzero_t
    INSTANTIATE_SCAN(SCAN_BINARY_OP, char, uint)
#else
    INSTANTIATE_SCAN_ALL(SCAN_BINARY_OP)
#endif
}