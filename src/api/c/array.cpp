/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <handle.hpp>
#include <ArrayInfo.hpp>
#include <platform.hpp>

const ArrayInfo&
getInfo(const af_array arr, bool check)
{
    const ArrayInfo *info = static_cast<ArrayInfo*>(reinterpret_cast<void *>(arr));

    if (check && info->getDevId() != detail::getActiveDeviceId()) {
        AF_ERROR("Input Array not created on current device", AF_ERR_DEVICE);
    }

    return *info;
}

af_err af_get_elements(dim_t *elems, const af_array arr)
{
    try {
        // Do not check for device mismatch
        *elems =  getInfo(arr, false).elements();
    } CATCHALL
    return AF_SUCCESS;
}

af_err af_get_type(af_dtype *type, const af_array arr)
{
    try {
        // Do not check for device mismatch
        *type = getInfo(arr, false).getType();
    } CATCHALL
    return AF_SUCCESS;
}

af_err af_get_dims(dim_t *d0, dim_t *d1, dim_t *d2, dim_t *d3,
                   const af_array in)
{
    try {
        // Do not check for device mismatch
        ArrayInfo info = getInfo(in, false);
        *d0 = info.dims()[0];
        *d1 = info.dims()[1];
        *d2 = info.dims()[2];
        *d3 = info.dims()[3];
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_get_numdims(unsigned *nd, const af_array in)
{
    try {
        // Do not check for device mismatch
        ArrayInfo info = getInfo(in, false);
        *nd = info.ndims();
    }
    CATCHALL
    return AF_SUCCESS;
}


#undef INSTANTIATE
#define INSTANTIATE(fn1, fn2)                           \
    af_err fn1(bool *result, const af_array in)         \
    {                                                   \
        try {                                           \
            ArrayInfo info = getInfo(in, false);   \
            *result = info.fn2();                       \
        }                                               \
        CATCHALL                                        \
            return AF_SUCCESS;                          \
    }

INSTANTIATE(af_is_empty       , isEmpty       )
INSTANTIATE(af_is_scalar      , isScalar      )
INSTANTIATE(af_is_row         , isRow         )
INSTANTIATE(af_is_column      , isColumn      )
INSTANTIATE(af_is_vector      , isVector      )
INSTANTIATE(af_is_complex     , isComplex     )
INSTANTIATE(af_is_real        , isReal        )
INSTANTIATE(af_is_double      , isDouble      )
INSTANTIATE(af_is_single      , isSingle      )
INSTANTIATE(af_is_realfloating, isRealFloating)
INSTANTIATE(af_is_floating    , isFloating    )
INSTANTIATE(af_is_integer     , isInteger     )
INSTANTIATE(af_is_bool        , isBool        )

#undef INSTANTIATE
