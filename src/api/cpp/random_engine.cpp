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

    randomEngine::randomEngine(const randomEngine& other)
    {
        if (this != &other) {
            AF_THROW(af_retain_random_engine(&engine, other.get()));
        }
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

    void randomEngine::setType(const randomType type)
    {
        AF_THROW(af_random_engine_set_type(&engine, type));
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
        AF_THROW(af_random_engine_set_seed(&engine, seed));
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

    array randu(const dim4 &dims, const af::dtype type)
    {
        af_array res;
        AF_THROW(af_randu(&res, dims.ndims(), dims.get(), type));
        return array(res);
    }

    array randu(const dim_t d0, const af::dtype ty)
    {
        return randu(dim4(d0), ty);
    }

    array randu(const dim_t d0,
                const dim_t d1, const af::dtype ty)
    {
        return randu(dim4(d0, d1), ty);
    }

    array randu(const dim_t d0,
                const dim_t d1, const dim_t d2, const af::dtype ty)
    {
        return randu(dim4(d0, d1, d2), ty);
    }

    array randu(const dim_t d0,
                const dim_t d1, const dim_t d2,
                const dim_t d3, const af::dtype ty)
    {
        return randu(dim4(d0, d1, d2, d3), ty);
    }

    array randn(const dim4 &dims, const af::dtype type)
    {
        af_array res;
        AF_THROW(af_randn(&res, dims.ndims(), dims.get(), type));
        return array(res);
    }

    array randn(const dim_t d0, const af::dtype ty)
    {
        return randn(dim4(d0), ty);
    }

    array randn(const dim_t d0,
                const dim_t d1, const af::dtype ty)
    {
        return randn(dim4(d0, d1), ty);
    }

    array randn(const dim_t d0,
                const dim_t d1, const dim_t d2, const af::dtype ty)
    {
        return randn(dim4(d0, d1, d2), ty);
    }

    array randn(const dim_t d0,
                const dim_t d1, const dim_t d2,
                const dim_t d3, const af::dtype ty)
    {
        return randn(dim4(d0, d1, d2, d3), ty);
    }

    void setDefaultRandomEngine(randomType rtype)
    {
        AF_THROW(af_set_default_random_engine(rtype));
    }

    void setSeed(const uintl seed)
    {
        AF_THROW(af_set_seed(seed));
    }

    uintl getSeed()
    {
        uintl seed = 0;
        AF_THROW(af_get_seed(&seed));
        return seed;
    }

}
