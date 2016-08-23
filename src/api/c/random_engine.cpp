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
#include <MersenneTwister.hpp>

using detail::cfloat;
using detail::cdouble;
using detail::uchar;
using detail::uniformDistribution;
using detail::normalDistribution;
using detail::initMersenneState;

using common::MaxBlocks;
using common::TableLength;
using common::MtStateLength;
using common::pos;
using common::sh1;
using common::sh2;
using common::mask;
using common::recursion_tbl;
using common::temper_tbl;

class RandomEngine
{
    public :
    af_random_type type;
    uintl seed;
    uintl counter;
    af_array pos;
    af_array sh1;
    af_array sh2;
    uint mask;
    af_array recursion_table;
    af_array temper_table;
    af_array state;
};

af_random_engine getRandomEngineHandle(const RandomEngine engine)
{
    RandomEngine *engineHandle = new RandomEngine;
    *engineHandle = engine;
    return static_cast<af_random_engine>(engineHandle);
}

RandomEngine* getRandomEngine(const af_random_engine engineHandle)
{
    return (RandomEngine *)engineHandle;
}

template<typename T>
static inline af_array uniformDistribution_(const af::dim4 &dims, RandomEngine *e)
{
    if (e->type == AF_RANDOM_MERSENNE) {
        return getHandle(uniformDistribution<T>(dims,
                    getArray<uint>(e->pos),
                    getArray<uint>(e->sh1),
                    getArray<uint>(e->sh2),
                    e->mask,
                    getArray<uint>(e->recursion_table),
                    getArray<uint>(e->temper_table),
                    getArray<uint>(e->state)));
    } else {
        return getHandle(uniformDistribution<T>(dims, e->type, e->seed, e->counter));
    }
}

template<typename T>
static inline af_array normalDistribution_(const af::dim4 &dims, RandomEngine *e)
{
    if (e->type == AF_RANDOM_MERSENNE) {
        return getHandle(normalDistribution<T>(dims,
                    getArray<uint>(e->pos),
                    getArray<uint>(e->sh1),
                    getArray<uint>(e->sh2),
                    e->mask,
                    getArray<uint>(e->recursion_table),
                    getArray<uint>(e->temper_table),
                    getArray<uint>(e->state)));
    } else {
        return getHandle(normalDistribution<T>(dims, e->type, e->seed, e->counter));
    }
}

af_err af_create_random_engine(af_random_engine *engineHandle, af_random_type rtype, uintl seed)
{
    try {
        RandomEngine e;
        e.type = rtype;
        e.seed = seed;
        e.counter = 0;
        if (rtype == AF_RANDOM_MERSENNE) {
            AF_CHECK(af_create_array(&e.pos, pos, 1, &MaxBlocks, u32));
            AF_CHECK(af_create_array(&e.sh1, sh1, 1, &MaxBlocks, u32));
            AF_CHECK(af_create_array(&e.sh2, sh2, 1, &MaxBlocks, u32));
            e.mask = mask;
            AF_CHECK(af_create_array(&e.recursion_table, recursion_tbl, 1, &TableLength, u32));
            AF_CHECK(af_create_array(&e.temper_table, temper_tbl, 1, &TableLength, u32));
            e.state = getHandle(initMersenneState(seed, getArray<uint>(e.recursion_table)));
        }
        *engineHandle = getRandomEngineHandle(e);
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_random_engine_set_seed(const uintl seed, af_random_engine engine)
{
    try {
        AF_CHECK(af_init());
        RandomEngine *e = getRandomEngine(engine);
        e->seed = seed;
        if (e->type == AF_RANDOM_MERSENNE) {
            initMersenneState(getWritableArray<uint>(e->state), seed, getArray<uint>(e->recursion_table));
        } else {
            e->counter = 0;
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_random_engine_get_seed(uintl * const seed, af_random_engine engine)
{
    try {
        AF_CHECK(af_init());
        RandomEngine *e = getRandomEngine(engine);
        *seed = e->seed;
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_random_engine_uniform(af_array *out, af_random_engine engine, const unsigned ndims, const dim_t * const dims, const af_dtype type)
{
    try {
        af_array result;
        AF_CHECK(af_init());

        af::dim4 d = verifyDims(ndims, dims);
        RandomEngine *e = getRandomEngine(engine);

        switch(type) {
        case f32:   result = uniformDistribution_<float  >(d, e); break;
        case c32:   result = uniformDistribution_<cfloat >(d, e); break;
        case f64:   result = uniformDistribution_<double >(d, e); break;
        case c64:   result = uniformDistribution_<cdouble>(d, e); break;
        case s32:   result = uniformDistribution_<int    >(d, e); break;
        case u32:   result = uniformDistribution_<uint   >(d, e); break;
        case s64:   result = uniformDistribution_<intl   >(d, e); break;
        case u64:   result = uniformDistribution_<uintl  >(d, e); break;
        case s16:   result = uniformDistribution_<short  >(d, e); break;
        case u16:   result = uniformDistribution_<ushort >(d, e); break;
        case u8:    result = uniformDistribution_<uchar  >(d, e); break;
        case b8:    result = uniformDistribution_<char   >(d, e); break;
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
        RandomEngine *e = getRandomEngine(engine);

        switch(type) {
        case f32:   result = normalDistribution_<float  >(d, e); break;
        case c32:   result = normalDistribution_<cfloat >(d, e); break;
        case f64:   result = normalDistribution_<double >(d, e); break;
        case c64:   result = normalDistribution_<cdouble>(d, e); break;
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
        RandomEngine *e = getRandomEngine(engineHandle);
        if (e->type == AF_RANDOM_MERSENNE) {
            AF_CHECK(af_release_array(e->pos));
            AF_CHECK(af_release_array(e->sh1));
            AF_CHECK(af_release_array(e->sh2));
            AF_CHECK(af_release_array(e->recursion_table));
            AF_CHECK(af_release_array(e->temper_table));
            AF_CHECK(af_release_array(e->state));
        }
        delete e;
    }
    CATCHALL;
    return AF_SUCCESS;
}
