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
#include <err_common.hpp>
#include <handle.hpp>
#include <random_engine.hpp>
#include <MersenneTwister.hpp>
#include <types.hpp>
#include <memory>

using namespace detail;
using namespace common;

class RandomEngine
{
    public :
    af_random_engine_type type;
    std::shared_ptr<uintl> seed;
    std::shared_ptr<uintl> counter;
    af_array pos;
    af_array sh1;
    af_array sh2;
    uint mask;
    af_array recursion_table;
    af_array temper_table;
    af_array state;

    RandomEngine(void) : type(AF_RANDOM_ENGINE_DEFAULT), seed(new uintl), counter(new uintl) {
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
        return getHandle(uniformDistribution<T>(dims,
                    getArray<uint>(e->pos),
                    getArray<uint>(e->sh1),
                    getArray<uint>(e->sh2),
                    e->mask,
                    getArray<uint>(e->recursion_table),
                    getArray<uint>(e->temper_table),
                    getArray<uint>(e->state)));
    } else {
        return getHandle(uniformDistribution<T>(dims, e->type, *(e->seed), *(e->counter)));
    }
}

template<typename T>
static inline af_array normalDistribution_(const af::dim4 &dims, RandomEngine *e)
{
    if (e->type == AF_RANDOM_ENGINE_MERSENNE_GP11213) {
        return getHandle(normalDistribution<T>(dims,
                    getArray<uint>(e->pos),
                    getArray<uint>(e->sh1),
                    getArray<uint>(e->sh2),
                    e->mask,
                    getArray<uint>(e->recursion_table),
                    getArray<uint>(e->temper_table),
                    getArray<uint>(e->state)));
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
    static RandomEngine re;
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
            AF_CHECK(af_create_array(&e.pos, pos, 1, &MaxBlocks, u32));
            AF_CHECK(af_create_array(&e.sh1, sh1, 1, &MaxBlocks, u32));
            AF_CHECK(af_create_array(&e.sh2, sh2, 1, &MaxBlocks, u32));
            e.mask = mask;
            AF_CHECK(af_create_array(&e.recursion_table, recursion_tbl, 1, &TableLength, u32));
            AF_CHECK(af_create_array(&e.temper_table, temper_tbl, 1, &TableLength, u32));
            AF_CHECK(af_create_handle(&e.state, 1, &MtStateLength, u32));
            initMersenneState(getWritableArray<uint>(e.state), seed, getArray<uint>(e.recursion_table));
        } else {
            dim_t empty = 0;
            AF_CHECK(af_create_handle(&e.pos, 1, &empty, u32));
            AF_CHECK(af_create_handle(&e.sh1, 1, &empty, u32));
            AF_CHECK(af_create_handle(&e.sh2, 1, &empty, u32));
            e.mask = 0;
            AF_CHECK(af_create_handle(&e.recursion_table, 1, &empty, u32));
            AF_CHECK(af_create_handle(&e.temper_table, 1, &empty, u32));
            AF_CHECK(af_create_handle(&e.state, 1, &empty, u32));
        }

        *engineHandle = getRandomEngineHandle(e);
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_retain_random_engine(af_random_engine *outHandle, const af_random_engine engineHandle)
{
    try {
        AF_CHECK(af_init());
        RandomEngine engine = *(getRandomEngine(engineHandle));
        RandomEngine out;

        out.type = engine.type;
        out.seed = engine.seed;
        out.counter = engine.counter;

        AF_CHECK(af_retain_array(&out.pos, engine.pos));
        AF_CHECK(af_retain_array(&out.sh1, engine.sh1));
        AF_CHECK(af_retain_array(&out.sh2, engine.sh2));
        out.mask = engine.mask;
        AF_CHECK(af_retain_array(&out.recursion_table, engine.recursion_table));
        AF_CHECK(af_retain_array(&out.temper_table, engine.temper_table));
        AF_CHECK(af_retain_array(&out.state, engine.state));

        *outHandle = getRandomEngineHandle(out);

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
                AF_CHECK(af_create_array(&e->pos, pos, 1, &MaxBlocks, u32));
                AF_CHECK(af_create_array(&e->sh1, sh1, 1, &MaxBlocks, u32));
                AF_CHECK(af_create_array(&e->sh2, sh2, 1, &MaxBlocks, u32));
                e->mask = mask;
                AF_CHECK(af_create_array(&e->recursion_table, recursion_tbl, 1, &TableLength, u32));
                AF_CHECK(af_create_array(&e->temper_table, temper_tbl, 1, &TableLength, u32));
                AF_CHECK(af_create_handle(&e->state, 1, &MtStateLength, u32));
                initMersenneState(getWritableArray<uint>(e->state), *(e->seed), getArray<uint>(e->recursion_table));
            } else if (e->type == AF_RANDOM_ENGINE_MERSENNE_GP11213) {
                AF_CHECK(af_release_array(e->pos));
                AF_CHECK(af_release_array(e->sh1));
                AF_CHECK(af_release_array(e->sh2));
                AF_CHECK(af_release_array(e->recursion_table));
                AF_CHECK(af_release_array(e->temper_table));
                AF_CHECK(af_release_array(e->state));
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
            initMersenneState(getWritableArray<uint>(e->state), seed, getArray<uint>(e->recursion_table));
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
        RandomEngine *e = getRandomEngine(engineHandle);
        if (e->type == AF_RANDOM_ENGINE_MERSENNE_GP11213) {
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
