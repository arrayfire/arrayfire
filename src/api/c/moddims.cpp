/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/data.h>
#include <backend.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <copy.hpp>

using af::dim4;
using namespace detail;

template<typename T>
Array<T> modDims(const Array<T>& in, const af::dim4 &newDims)
{
    //FIXME: Figure out a better way
    evalArray<T>(in);

    Array<T> Out = in;

    if (!in.isOwner()) {
        Out = copyArray<T>(in);
    }
    Out.modDims(newDims);

    return Out;
}

af_err af_moddims(af_array *out, const af_array in,
                  const unsigned ndims, const dim_t * const dims)
{
    try {
        ARG_ASSERT(2, ndims >= 1);
        ARG_ASSERT(3, dims != NULL);

        af_array output = 0;
        dim4 newDims(ndims, dims);
        ArrayInfo info = getInfo(in);
        dim_t in_elements = info.elements();
        dim_t new_elements = newDims.elements();

        DIM_ASSERT(1, in_elements == new_elements);

        af_dtype type = info.getType();

        switch(type) {
        case f32: output = getHandle(modDims<float  >(getArray<float  >(in), newDims)); break;
        case c32: output = getHandle(modDims<cfloat >(getArray<cfloat >(in), newDims)); break;
        case f64: output = getHandle(modDims<double >(getArray<double >(in), newDims)); break;
        case c64: output = getHandle(modDims<cdouble>(getArray<cdouble>(in), newDims)); break;
        case b8:  output = getHandle(modDims<char   >(getArray<char   >(in), newDims)); break;
        case s32: output = getHandle(modDims<int    >(getArray<int    >(in), newDims)); break;
        case u32: output = getHandle(modDims<uint   >(getArray<uint   >(in), newDims)); break;
        case u8:  output = getHandle(modDims<uchar  >(getArray<uchar  >(in), newDims)); break;
        case s64: output = getHandle(modDims<intl   >(getArray<intl   >(in), newDims)); break;
        case u64: output = getHandle(modDims<uintl  >(getArray<uintl  >(in), newDims)); break;
        default: TYPE_ERROR(1, type);
        }
        std::swap(*out,output);
    }
    CATCHALL

    return AF_SUCCESS;
}

af_err af_flat(af_array *out, const af_array in)
{
    af_array res;
    try {

        ArrayInfo in_info = getInfo(in);

        if (in_info.ndims() == 1) {
            AF_CHECK(af_retain_array(&res, in));
        } else {
            const dim_t num = (dim_t)(in_info.elements());
            AF_CHECK(af_moddims(&res, in, 1, &num));
        }

        std::swap(*out, res);
    } CATCHALL;
    return AF_SUCCESS;
}
