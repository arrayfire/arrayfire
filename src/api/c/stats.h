/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

template<typename T, typename Other>
struct is_same{
    static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
    static const bool value = true;
};

template<bool, typename T, typename O>
struct cond_type;

template<typename T, typename Other>
struct cond_type<true, T, Other> {
    typedef T type;
};

template<typename T, typename Other>
struct cond_type<false, T, Other> {
    typedef Other type;
};

template<typename T>
struct baseOutType {
    typedef typename cond_type< is_same<T, cdouble>::value ||
                                is_same<T, double>::value,
                                double,
                                float>::type type;
};

template<typename Ti, typename To>
inline To mean(const Array<Ti>& in)
{
    To out    = reduce_all<af_add_t, Ti, To>(in);
    To result = division(out, in.elements());
    return result;
}

template<typename T, typename Tw>
static T mean(const Array<T>& input, const Array<Tw>& weights)
{
    dim4 iDims = input.dims();

    Array<T> wtdInput = arithOp<T, af_mul_t>(input, weights, iDims);

    T wtdSum = reduce_all<af_add_t, T, T>(wtdInput);
    T wtsSum = reduce_all<af_add_t, T, T>(weights);

    return division(wtdSum, wtsSum);
}

#define COMPLEX_TYPE_SPECILIZATION(T, Tw) \
template<>\
T mean<T, Tw>(const Array<T>& input, const Array<Tw>& weights)\
{\
    Array<T> wts = cast<T, Tw>(weights);\
    dim4 iDims   = input.dims();\
    Array<T> wtdInput = arithOp<T, af_mul_t>(input, wts, iDims);\
    T wtdSum  = reduce_all<af_add_t, T, T>(wtdInput);\
    Tw wtsSum = reduce_all<af_add_t, Tw, Tw>(weights);\
    return division(wtdSum, wtsSum);\
}

COMPLEX_TYPE_SPECILIZATION(cfloat, float)
COMPLEX_TYPE_SPECILIZATION(cdouble, double)

template<typename Ti, typename To>
inline Array<To> mean(const Array<Ti>& in, dim_t dim)
{
    Array<To> redArr = reduce<af_add_t, Ti, To>(in, dim);

    dim4 iDims = in.dims();
    dim4 oDims = redArr.dims();

    Array<To> cnstArr = createValueArray<To>(oDims, scalar<To>(iDims[dim]));
    Array<To> result  = arithOp<To, af_div_t>(redArr, cnstArr, oDims);

    return result;
}

template<typename T>
inline Array<T> mean(const Array<T>& in, const Array<T>& wts, dim_t dim)
{
    dim4 iDims = in.dims();

    Array<T> wtdInput = arithOp<T, af_mul_t>(in, wts, iDims);
    Array<T> redArr   = reduce<af_add_t, T, T>(wtdInput, dim);
    Array<T> wtsSum   = reduce<af_add_t, T, T>(wts, dim);

    dim4 oDims = redArr.dims();

    Array<T> result = arithOp<T, af_div_t>(redArr, wtsSum, oDims);

    return result;
}
