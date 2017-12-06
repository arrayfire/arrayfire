/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/random.h>

#include <af/array.h>
#include <af/data.h>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <backend.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <random_engine.hpp>
#include <common/MersenneTwister.hpp>
#include <types.hpp>
#include <memory>

using namespace detail;
using namespace common;

using af::dim4;

Array<uint> emptyArray()
{
    static const Array<uint> EMPTY_ARRAY = createEmptyArray<uint>(af::dim4(0));

    return EMPTY_ARRAY;
}

struct RandomEngine
{
    af_random_engine_type type;
    std::shared_ptr<uintl> seed;
    std::shared_ptr<uintl> counter;
    Array<uint> pos;
    Array<uint> sh1;
    Array<uint> sh2;
    uint mask;
    Array<uint> recursion_table;
    Array<uint> temper_table;
    Array<uint> state;

    RandomEngine(void)
        : type(AF_RANDOM_ENGINE_DEFAULT), seed(new uintl), counter(new uintl),
        pos(emptyArray()), sh1(emptyArray()), sh2(emptyArray()), mask(0),
        recursion_table(emptyArray()), temper_table(emptyArray()), state(emptyArray())
    {
        *seed = 0;
        *counter = 0;
    }
};

af_random_engine getRandomEngineHandle(const RandomEngine engine)
{
    RandomEngine *engineHandle = new RandomEngine;
    *engineHandle = engine;
    return static_cast<af_random_engine>(engineHandle);
}

RandomEngine* getRandomEngine(const af_random_engine engineHandle)
{
    if (engineHandle == 0) {
        AF_ERROR("Uninitialized random engine", AF_ERR_ARG);
    }
    return (RandomEngine *)engineHandle;
}

template<typename T>
static inline af_array uniformDistribution_(const af::dim4 &dims, RandomEngine *e)
{
    if (e->type == AF_RANDOM_ENGINE_MERSENNE_GP11213) {
        return getHandle(uniformDistribution<T>(dims, e->pos, e->sh1, e->sh2, e->mask,
                    e->recursion_table, e->temper_table, e->state));
    } else {
        return getHandle(uniformDistribution<T>(dims, e->type, *(e->seed), *(e->counter)));
    }
}

template<typename T>
static inline af_array normalDistribution_(const af::dim4 &dims, RandomEngine *e)
{
    if (e->type == AF_RANDOM_ENGINE_MERSENNE_GP11213) {
        return getHandle(normalDistribution<T>(dims, e->pos, e->sh1, e->sh2, e->mask,
                    e->recursion_table, e->temper_table, e->state));
    } else {
        return getHandle(normalDistribution<T>(dims, e->type, *(e->seed), *(e->counter)));
    }
}

static void validateRandomType(const af_random_engine_type type)
{
    if ((type != AF_RANDOM_ENGINE_PHILOX_4X32_10)
    &&  (type != AF_RANDOM_ENGINE_THREEFRY_2X32_16)
    &&  (type != AF_RANDOM_ENGINE_MERSENNE_GP11213)
    &&  (type != AF_RANDOM_ENGINE_PHILOX)
    &&  (type != AF_RANDOM_ENGINE_THREEFRY)
    &&  (type != AF_RANDOM_ENGINE_MERSENNE)
    &&  (type != AF_RANDOM_ENGINE_DEFAULT)) {
        AF_ERROR("Invalid random type", AF_ERR_ARG);
    }
}

af_err af_get_default_random_engine(af_random_engine *r)
{
    AF_CHECK(af_init());

    thread_local RandomEngine re;
    *r = static_cast<af_random_engine> (&re);
    return AF_SUCCESS;
}

af_err af_create_random_engine(af_random_engine *engineHandle, af_random_engine_type rtype, uintl seed)
{
    try {
        AF_CHECK(af_init());
        validateRandomType(rtype);

        RandomEngine e;
        e.type = rtype;
        *e.seed = seed;
        *e.counter = 0;

        if (rtype == AF_RANDOM_ENGINE_MERSENNE_GP11213) {
            e.pos  = createHostDataArray<uint>(af::dim4(MaxBlocks), pos);
            e.sh1  = createHostDataArray<uint>(af::dim4(MaxBlocks), sh1);
            e.sh2  = createHostDataArray<uint>(af::dim4(MaxBlocks), sh2);
            e.mask = mask;

            e.recursion_table = createHostDataArray<uint>(af::dim4(TableLength), recursion_tbl);
            e.temper_table    = createHostDataArray<uint>(af::dim4(TableLength), temper_tbl);
            e.state           = createEmptyArray<uint>(af::dim4(MtStateLength));

            initMersenneState(e.state, seed, e.recursion_table);
        }

        *engineHandle = getRandomEngineHandle(e);
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_retain_random_engine(af_random_engine *outHandle, const af_random_engine engineHandle)
{
    try {
        AF_CHECK(af_init());
        *outHandle = getRandomEngineHandle(*(getRandomEngine(engineHandle)));
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_random_engine_set_type(af_random_engine *engine, const af_random_engine_type rtype)
{
    try {
        AF_CHECK(af_init());
        validateRandomType(rtype);
        RandomEngine *e = getRandomEngine(*engine);
        if (rtype != e->type) {
            if (rtype == AF_RANDOM_ENGINE_MERSENNE_GP11213) {
                e->pos  = createHostDataArray<uint>(af::dim4(MaxBlocks), pos);
                e->sh1  = createHostDataArray<uint>(af::dim4(MaxBlocks), sh1);
                e->sh2  = createHostDataArray<uint>(af::dim4(MaxBlocks), sh2);
                e->mask = mask;

                e->recursion_table = createHostDataArray<uint>(af::dim4(TableLength), recursion_tbl);
                e->temper_table    = createHostDataArray<uint>(af::dim4(TableLength), temper_tbl);
                e->state           = createEmptyArray<uint>(af::dim4(MtStateLength));

                initMersenneState(e->state, *(e->seed), e->recursion_table);
            } else if (e->type == AF_RANDOM_ENGINE_MERSENNE_GP11213) {
                e->pos  = emptyArray();
                e->sh1  = emptyArray();
                e->sh2  = emptyArray();
                e->mask = 0;
                e->recursion_table = emptyArray();
                e->temper_table    = emptyArray();
                e->state           = emptyArray();
            }
            e->type = rtype;
        }
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_random_engine_get_type(af_random_engine_type *rtype, const af_random_engine engine)
{
    try {
        AF_CHECK(af_init());
        RandomEngine *e = getRandomEngine(engine);
        *rtype = e->type;
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_set_default_random_engine_type(const af_random_engine_type rtype)
{
    try {
        AF_CHECK(af_init());
        af_random_engine e;
        AF_CHECK(af_get_default_random_engine(&e));
        AF_CHECK(af_random_engine_set_type(&e, rtype));
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_random_engine_set_seed(af_random_engine *engine, const uintl seed)
{
    try {
        AF_CHECK(af_init());
        RandomEngine *e = getRandomEngine(*engine);
        *(e->seed) = seed;
        if (e->type == AF_RANDOM_ENGINE_MERSENNE_GP11213) {
            initMersenneState(e->state, seed, e->recursion_table);
        } else {
            *(e->counter) = 0;
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
        *seed = *(e->seed);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_random_uniform(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type, af_random_engine engine)
{
    try {
        AF_CHECK(af_init());
        af_array result;

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

af_err af_random_normal(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type, af_random_engine engine)
{
    try {
        AF_CHECK(af_init());
        af_array result;

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
        AF_CHECK(af_init());
        delete getRandomEngine(engineHandle);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_randu(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type)
{
    try {
        AF_CHECK(af_init());
        af_array result;

        af_random_engine engine;
        AF_CHECK(af_get_default_random_engine(&engine));
        RandomEngine *e = getRandomEngine(engine);
        af::dim4 d = verifyDims(ndims, dims);

        switch(type) {
        case f32: result = uniformDistribution_<float  >(d, e); break;
        case c32: result = uniformDistribution_<cfloat >(d, e); break;
        case f64: result = uniformDistribution_<double >(d, e); break;
        case c64: result = uniformDistribution_<cdouble>(d, e); break;
        case s32: result = uniformDistribution_<int    >(d, e); break;
        case u32: result = uniformDistribution_<uint   >(d, e); break;
        case s64: result = uniformDistribution_<intl   >(d, e); break;
        case u64: result = uniformDistribution_<uintl  >(d, e); break;
        case s16: result = uniformDistribution_<short  >(d, e); break;
        case u16: result = uniformDistribution_<ushort >(d, e); break;
        case u8:  result = uniformDistribution_<uchar  >(d, e); break;
        case b8:  result = uniformDistribution_<char   >(d, e); break;
        default:  TYPE_ERROR(3, type);
        }
        std::swap(*out, result);
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_randn(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type)
{
    try {
        AF_CHECK(af_init());
        af_array result;

        af_random_engine engine;
        AF_CHECK(af_get_default_random_engine(&engine));
        RandomEngine *e = getRandomEngine(engine);
        af::dim4 d = verifyDims(ndims, dims);

        switch(type) {
        case f32: result = normalDistribution_<float  >(d, e); break;
        case c32: result = normalDistribution_<cfloat >(d, e); break;
        case f64: result = normalDistribution_<double >(d, e); break;
        case c64: result = normalDistribution_<cdouble>(d, e); break;
        default:  TYPE_ERROR(3, type);
        }
        std::swap(*out, result);
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_set_seed(const uintl seed)
{
    try {
        AF_CHECK(af_init());
        af_random_engine engine;
        AF_CHECK(af_get_default_random_engine(&engine));
        AF_CHECK(af_random_engine_set_seed(&engine, seed));
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_seed(uintl *seed)
{
    try {
        AF_CHECK(af_init());
        af_random_engine e;
        AF_CHECK(af_get_default_random_engine(&e));
        AF_CHECK(af_random_engine_get_seed(seed, e));
    } CATCHALL;
    return AF_SUCCESS;
}
