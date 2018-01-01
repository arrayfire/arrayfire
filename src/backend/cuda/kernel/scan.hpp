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
}
