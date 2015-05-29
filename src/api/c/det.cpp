/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/lapack.h>
#include <af/defines.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <math.hpp>
#include <lu.hpp>
#include <diagonal.hpp>
#include <copy.hpp>

using af::dim4;
using namespace detail;

template<typename T>
T det(const af_array a)
{
    const Array<T> A = getArray<T>(a);

    const int num = A.dims()[0];

    std::vector<T> hD(num);
    std::vector<int> hP(num);

    Array<T> D = createEmptyArray<T>(dim4());
    Array<int> pivot = createEmptyArray<int>(dim4());

    // Free memory as soon as possible
    {
        Array<T> A_copy = copyArray<T>(A);

        Array<int> pivot = lu_inplace(A_copy, false);
        copyData(&hP[0], pivot);

        Array<T> D = diagExtract(A_copy, 0);
        copyData(&hD[0], D);
    }

    bool is_neg = false;
    T res = scalar<T>(is_neg ? -1 : 1);
    for (int i = 0; i < num; i++) {
        res = res * hD[i];
        is_neg ^= (hP[i] != (i+1));
    }

    if (is_neg) res = res * scalar<T>(-1);

    return res;
}

af_err af_det(double *real_val, double *imag_val, const af_array in)
{

    try {
        ArrayInfo i_info = getInfo(in);

        if (i_info.ndims() > 2) {
            AF_ERROR("solve can not be used in batch mode", AF_ERR_BATCH);
        }

        af_dtype type = i_info.getType();

        DIM_ASSERT(1, i_info.dims()[0] == i_info.dims()[1]);      // Only square matrices
        ARG_ASSERT(1, i_info.isFloating());                       // Only floating and complex types

        *real_val = 0;
        *imag_val = 0;

        cfloat cfval;
        cdouble cdval;

        switch(type) {
        case f32: *real_val = det<float  >(in);  break;
        case f64: *real_val = det<double >(in);  break;
        case c32:
            cfval = det<cfloat >(in);
            *real_val = real(cfval);
            *imag_val = imag(cfval);
            break;
        case c64:
            cdval = det<cdouble>(in);
            *real_val = real(cdval);
            *imag_val = imag(cdval);
            break;
        default:  TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}
