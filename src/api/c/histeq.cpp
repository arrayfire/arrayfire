/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/image.h>
#include <af/index.h>
#include <af/defines.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <cast.hpp>
#include <scan.hpp>
#include <arith.hpp>
#include <reduce.hpp>
#include <ArrayIndex.hpp>

using namespace detail;

template<typename T, typename hType>
static af_array histequal(const af_array& in, const af_array& hist)
{
    const Array<T> input = getArray<T>(in);

    af_array vInput = 0;
    AF_CHECK(af_flat(&vInput, in));

    Array<float>* fHist  = cast<float>(getArray<hType>(hist));

    dim4 hDims = fHist->dims();
    dim_type grayLevels = fHist->elements();

    Array<float>* cdf = detail::scan<af_add_t, float, float>(*fHist, 0);

    float minCdf = detail::reduce_all<af_min_t, float, float>(*cdf);
    float maxCdf = detail::reduce_all<af_max_t, float, float>(*cdf);
    float factor = (float)(grayLevels-1)/(maxCdf - minCdf);

    // constant array of min value from cdf
    Array<float>* minCnst = createValueArray<float>(hDims, minCdf);
    // constant array of factor variable
    Array<float>* facCnst = createValueArray<float>(hDims, factor);
    // cdf(i) - min for all elements
    Array<float>* diff    = detail::arithOp<float, af_sub_t>(*cdf, *minCnst, hDims);
    // multiply factor with difference
    Array<float>* normCdf = detail::arithOp<float, af_mul_t>(*diff, *facCnst, hDims);
    // index input array with normalized cdf array
    Array<float>* idxArr  = detail::arrayIndex<float, T>(*normCdf, getArray<T>(vInput), 0);

    Array<T>* result = cast<T>(*idxArr);

    destroyArray<float>(*idxArr);
    destroyArray<float>(*normCdf);
    destroyArray<float>(*diff);
    destroyArray<float>(*facCnst);
    destroyArray<float>(*minCnst);
    destroyArray<float>(*cdf);
    destroyArray<float>(*fHist);
    AF_CHECK(af_destroy_array(vInput));

    return getHandle<T>(*result);
}

af_err af_histequal(af_array *out, const af_array in, const af_array hist)
{
    try {
        ArrayInfo dataInfo = getInfo(in);
        ArrayInfo histInfo = getInfo(hist);

        af_dtype dataType  = dataInfo.getType();
        af::dim4 histDims  = histInfo.dims();

        ARG_ASSERT(2, (histDims.ndims()==1));

        af_array output = 0;
        switch(dataType) {
            case f64: output = histequal<double, uint>(in, hist); break;
            case f32: output = histequal<float , uint>(in, hist); break;
            case s32: output = histequal<int   , uint>(in, hist); break;
            case u32: output = histequal<uint  , uint>(in, hist); break;
            case u8 : output = histequal<uchar , uint>(in, hist); break;
            default : TYPE_ERROR(1, dataType);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
