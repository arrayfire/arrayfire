/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/data.h>
#include <af/dim4.hpp>
#include <af/random.h>
#include "error.hpp"

namespace af {
randomEngine::randomEngine(randomEngineType type, unsigned long long seed)
    : engine(0) {
    AF_THROW(af_create_random_engine(&engine, type, seed));
}

randomEngine::randomEngine(const randomEngine &other) : engine(0) {
    if (this != &other) {
        AF_THROW(af_retain_random_engine(&engine, other.get()));
    }
}

randomEngine::randomEngine(af_random_engine engine) : engine(engine) {}

randomEngine::~randomEngine() {
    if (engine) { af_release_random_engine(engine); }
}

randomEngine &randomEngine::operator=(const randomEngine &other) {
    if (this != &other) {
        AF_THROW(af_release_random_engine(engine));
        AF_THROW(af_retain_random_engine(&engine, other.get()));
    }
    return *this;
}

randomEngineType randomEngine::getType() {
    af_random_engine_type type;
    AF_THROW(af_random_engine_get_type(&type, engine));
    return type;
}

void randomEngine::setType(const randomEngineType type) {
    AF_THROW(af_random_engine_set_type(&engine, type));
}

void randomEngine::setSeed(const unsigned long long seed) {
    AF_THROW(af_random_engine_set_seed(&engine, seed));
}

unsigned long long randomEngine::getSeed() const {
    unsigned long long seed;
    AF_THROW(af_random_engine_get_seed(&seed, engine));
    return seed;
}

af_random_engine randomEngine::get() const { return engine; }

array randu(const dim4 &dims, const dtype ty, randomEngine &r) {
    af_array out;
    AF_THROW(af_random_uniform(&out, dims.ndims(), dims.get(), ty, r.get()));
    return array(out);
}

array randn(const dim4 &dims, const dtype ty, randomEngine &r) {
    af_array out;
    AF_THROW(af_random_normal(&out, dims.ndims(), dims.get(), ty, r.get()));
    return array(out);
}

array randu(const dim4 &dims, const af::dtype type) {
    af_array res;
    AF_THROW(af_randu(&res, dims.ndims(), dims.get(), type));
    return array(res);
}

array randu(const dim_t d0, const af::dtype ty) { return randu(dim4(d0), ty); }

array randu(const dim_t d0, const dim_t d1, const af::dtype ty) {
    return randu(dim4(d0, d1), ty);
}

array randu(const dim_t d0, const dim_t d1, const dim_t d2,
            const af::dtype ty) {
    return randu(dim4(d0, d1, d2), ty);
}

array randu(const dim_t d0, const dim_t d1, const dim_t d2, const dim_t d3,
            const af::dtype ty) {
    return randu(dim4(d0, d1, d2, d3), ty);
}

array randn(const dim4 &dims, const af::dtype type) {
    af_array res;
    AF_THROW(af_randn(&res, dims.ndims(), dims.get(), type));
    return array(res);
}

array randn(const dim_t d0, const af::dtype ty) { return randn(dim4(d0), ty); }

array randn(const dim_t d0, const dim_t d1, const af::dtype ty) {
    return randn(dim4(d0, d1), ty);
}

array randn(const dim_t d0, const dim_t d1, const dim_t d2,
            const af::dtype ty) {
    return randn(dim4(d0, d1, d2), ty);
}

array randn(const dim_t d0, const dim_t d1, const dim_t d2, const dim_t d3,
            const af::dtype ty) {
    return randn(dim4(d0, d1, d2, d3), ty);
}

void setDefaultRandomEngineType(randomEngineType rtype) {
    AF_THROW(af_set_default_random_engine_type(rtype));
}

randomEngine getDefaultRandomEngine() {
    af_random_engine internal_handle = 0;
    af_random_engine handle          = 0;
    AF_THROW(af_get_default_random_engine(&internal_handle));
    AF_THROW(af_retain_random_engine(&handle, internal_handle));
    return randomEngine(handle);
}

void setSeed(const unsigned long long seed) { AF_THROW(af_set_seed(seed)); }

unsigned long long getSeed() {
    unsigned long long seed = 0;
    AF_THROW(af_get_seed(&seed));
    return seed;
}

}  // namespace af
