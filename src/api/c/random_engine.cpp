/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/random_engine.h>

#include <af/array.h>
#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <backend.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <random_engine.hpp>
#include <iostream>

using detail::cfloat;
using detail::cdouble;
using detail::uchar;
using detail::uniformDistribution;
using detail::normalDistribution;

typedef struct {
    af_random_type type;
    unsigned long long seed;
    unsigned long long counter;
} af_random_engine_t;

//TODO : static pointer to random_engine_t that wil hold the default random engine
//protect this object with mutex

af_random_engine getRandomEngineHandle(const af_random_engine_t engine)
{
    af_random_engine_t *engineHandle = new af_random_engine_t;
    *engineHandle = engine;
    return static_cast<af_random_engine>(engineHandle);
}

af_random_engine_t* getRandomEngine(const af_random_engine engineHandle)
{
    return (af_random_engine_t *)engineHandle;
}

template<typename T>
static inline af_array uniformDistribution_(const af::dim4 &dims,
        const af_random_type type, const unsigned long long seed, unsigned long long &counter)
{
    return getHandle(uniformDistribution<T>(dims, type, seed, counter));
}

template<typename T>
static inline af_array normalDistribution_(const af::dim4 &dims,
        const af_random_type type, const unsigned long long seed, unsigned long long &counter)
{
    return getHandle(normalDistribution<T>(dims, type, seed, counter));
}

af_err af_create_random_engine(af_random_engine *engineHandle, af_random_type rtype, unsigned long long seed)
{
    try {
        af_random_engine_t engine{rtype, seed, 0};
        *engineHandle = getRandomEngineHandle(engine);
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_random_engine_uniform(af_array *out, af_random_engine engine, const unsigned ndims, const dim_t * const dims, const af_dtype type)
{
    try {
        af_array result;
        AF_CHECK(af_init());

        af::dim4 d = verifyDims(ndims, dims);
        af_random_engine_t *e = getRandomEngine(engine);

        switch(type) {
        case f32:   result = uniformDistribution_<float  >(d, e->type, e->seed, e->counter);    break;
        case c32:   result = uniformDistribution_<cfloat >(d, e->type, e->seed, e->counter);    break;
        case f64:   result = uniformDistribution_<double >(d, e->type, e->seed, e->counter);    break;
        case c64:   result = uniformDistribution_<cdouble>(d, e->type, e->seed, e->counter);    break;
        case s32:   result = uniformDistribution_<int    >(d, e->type, e->seed, e->counter);    break;
        case u32:   result = uniformDistribution_<uint   >(d, e->type, e->seed, e->counter);    break;
        case s64:   result = uniformDistribution_<intl   >(d, e->type, e->seed, e->counter);    break;
        case u64:   result = uniformDistribution_<uintl  >(d, e->type, e->seed, e->counter);    break;
        case s16:   result = uniformDistribution_<short  >(d, e->type, e->seed, e->counter);    break;
        case u16:   result = uniformDistribution_<ushort >(d, e->type, e->seed, e->counter);    break;
        case u8:    result = uniformDistribution_<uchar  >(d, e->type, e->seed, e->counter);    break;
        case b8:    result = uniformDistribution_<char   >(d, e->type, e->seed, e->counter);    break;
        default:    TYPE_ERROR(4, type);
        }
        std::swap(*out, result);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_random_engine_normal(af_array *out, af_random_engine engine, const unsigned ndims, const dim_t * const dims, const af_dtype type)
{
    try {
        af_array result;
        AF_CHECK(af_init());

        af::dim4 d = verifyDims(ndims, dims);
        af_random_engine_t *e = getRandomEngine(engine);

        switch(type) {
        case f32:   result = normalDistribution_<float  >(d, e->type, e->seed, e->counter);    break;
        case c32:   result = normalDistribution_<cfloat >(d, e->type, e->seed, e->counter);    break;
        case f64:   result = normalDistribution_<double >(d, e->type, e->seed, e->counter);    break;
        case c64:   result = normalDistribution_<cdouble>(d, e->type, e->seed, e->counter);    break;
        default:    TYPE_ERROR(4, type);
        }
        std::swap(*out, result);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_release_random_engine(af_random_engine engineHandle)
{
    try {
        delete (af_random_engine_t *)engineHandle;
    }
    CATCHALL;
    return AF_SUCCESS;
}
