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
#include <af/index.h>
#include <backend.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <copy.hpp>

using af::dim4;
using namespace detail;

template<typename T>
af_array modDims(const af_array in, const af::dim4 &newDims)
{
    Array<T> *out = copyArray(getArray<T>(in));
    out->modDims(newDims);
    return getHandle(*out);
}

af_err af_moddims(af_array *out, const af_array in, const unsigned ndims, const dim_type * const dims)
{
    if (ndims<1 || dims==NULL) {
        return AF_ERR_ARG;
    }

    af_err ret = AF_ERR_RUNTIME;
    af_array output = 0;

    try {
        dim4 newDims((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            newDims[i] = dims[i];
        }

        ArrayInfo info = getInfo(in);

        if (info.elements() != (size_t)newDims.elements()) {
            return AF_ERR_SIZE;
        }

        af_dtype type = info.getType();
        af_get_type(&type, in);

        switch(type) {
            case f32: output = modDims<float  >(in, newDims); break;
            case c32: output = modDims<cfloat >(in, newDims); break;
            case f64: output = modDims<double >(in, newDims); break;
            case c64: output = modDims<cdouble>(in, newDims); break;
            case b8:  output = modDims<char   >(in, newDims); break;
            case s32: output = modDims<int    >(in, newDims); break;
            case u32: output = modDims<uint   >(in, newDims); break;
            case u8:  output = modDims<uchar  >(in, newDims); break;
            case s8:  output = modDims<char>(in,newDims);     break;
            default:  ret = AF_ERR_NOT_SUPPORTED;             break;
        }
        if (ret!=AF_ERR_NOT_SUPPORTED) {
            std::swap(*out,output);
            ret = AF_SUCCESS;
        }
    }
    CATCHALL

    return ret;
}
