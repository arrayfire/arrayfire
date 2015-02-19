/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

template<typename T>
using baseOutType = typename std::conditional<  std::is_same<T, cdouble>::value ||
                                                std::is_same<T, double>::value,
                                              double,
                                              float>::type;
template<typename T>
inline T mean(const Array<T>& in)
{
    return division(detail::reduce_all<af_add_t, T, T>(in), in.elements());
}

template<typename T, typename wType>
inline T mean(const Array<T>& in, const Array<wType>& weights)
{
    Array<T> *wts   = cast<T>(weights);

    dim4 iDims = in.dims();

    Array<T>* wtdInput = detail::arithOp<T, af_mul_t>(in, *wts, iDims);

    T wtdSum = detail::reduce_all<af_add_t, T, T>(*wtdInput);
    wType wtsSum = detail::reduce_all<af_add_t, wType, wType>(weights);

    destroyArray<T>(*wtdInput);
    destroyArray<T>(*wts);

    return division(wtdSum, wtsSum);
}

template<typename T>
inline Array<T>* mean(const Array<T>& in, dim_type dim)
{
    Array<T>* redArr = reduce<af_add_t, T, T>(in, dim);

    dim4 iDims = in.dims();
    dim4 oDims = redArr->dims();

    Array<T> *cnstArr = createValueArray<T>(oDims, scalar<T>(iDims[dim]));
    Array<T> *result  = detail::arithOp<T, af_div_t>(*redArr, *cnstArr, oDims);

    destroyArray<T>(*cnstArr);
    destroyArray<T>(*redArr);

    return result;
}

template<typename T>
inline Array<T>* mean(const Array<T>& in, const Array<T>& wts, dim_type dim)
{
    dim4 iDims = in.dims();

    Array<T>* wtdInput = detail::arithOp<T, af_mul_t>(in, wts, iDims);
    Array<T>* redArr   = reduce<af_add_t, T, T>(*wtdInput, dim);
    Array<T>* wtsSum   = reduce<af_add_t, T, T>(wts, dim);

    dim4 oDims = redArr->dims();

    Array<T> *result = detail::arithOp<T, af_div_t>(*redArr, *wtsSum, oDims);

    destroyArray<T>(*wtsSum);
    destroyArray<T>(*redArr);
    destroyArray<T>(*wtdInput);

    return result;
}
