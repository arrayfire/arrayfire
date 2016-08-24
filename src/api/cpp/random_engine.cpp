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
#include <af/dim4.hpp>
#include <af/data.h>
#include "error.hpp"

namespace af
{
    randomEngine::randomEngine(randomType type, uintl seed)
    {
        AF_THROW(af_create_random_engine(&engine, type, seed));
    }

    randomEngine::~randomEngine()
    {
        if (engine) {
            af_release_random_engine(engine);
        }
    }

    randomEngine& randomEngine::operator= (const randomEngine& other)
    {
        if (this != &other) {
            AF_THROW(af_release_random_engine(engine));
            AF_THROW(af_retain_random_engine(&engine, other.get()));
        }
        return *this;
    }

    array randomEngine::uniform(const dim_t dim0, const dtype ty)
    {
        dim4 d(dim0, 1, 1, 1);
        return uniform(d, ty);
    }

    array randomEngine::uniform(const dim_t dim0, const dim_t dim1, const dtype ty)
    {
        dim4 d(dim0, dim1, 1, 1);
        return uniform(d, ty);
    }

    array randomEngine::uniform(const dim_t dim0, const dim_t dim1, const dim_t dim2, const dtype ty)
    {
        dim4 d(dim0, dim1, dim2, 1);
        return uniform(d, ty);
    }

    array randomEngine::uniform(const dim_t dim0, const dim_t dim1, const dim_t dim2, const dim_t dim3, const dtype ty)
    {
        dim4 d(dim0, dim1, dim2, dim3);
        return uniform(d, ty);
    }

    array randomEngine::uniform(const dim4& dims, const dtype ty)
    {
        af_array out;
        AF_THROW(af_random_engine_uniform(&out, engine, dims.ndims(), dims.get(), ty));
        return array(out);
    }

    array randomEngine::normal(const dim_t dim0, const dtype ty)
    {
        dim4 d(dim0, 1, 1, 1);
        return normal(d, ty);
    }

    array randomEngine::normal(const dim_t dim0, const dim_t dim1, const dtype ty)
    {
        dim4 d(dim0, dim1, 1, 1);
        return normal(d, ty);
    }

    array randomEngine::normal(const dim_t dim0, const dim_t dim1, const dim_t dim2, const dtype ty)
    {
        dim4 d(dim0, dim1, dim2, 1);
        return normal(d, ty);
    }

    array randomEngine::normal(const dim_t dim0, const dim_t dim1, const dim_t dim2, const dim_t dim3, const dtype ty)
    {
        dim4 d(dim0, dim1, dim2, dim3);
        return normal(d, ty);
    }

    array randomEngine::normal(const dim4& dims, const dtype ty)
    {
        af_array out;
        AF_THROW(af_random_engine_normal(&out, engine, dims.ndims(), dims.get(), ty));
        return array(out);
    }

    void randomEngine::setSeed(const uintl seed)
    {
        AF_THROW(af_random_engine_set_seed(seed, engine));
    }

    uintl randomEngine::getSeed(void) const
    {
        uintl seed;
        AF_THROW(af_random_engine_get_seed(&seed, engine));
        return seed;
    }

    af_random_engine randomEngine::get() const
    {
        return engine;
    }

}
