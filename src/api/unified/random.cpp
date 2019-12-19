/*******************************************************
 * Copyright(c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/random.h>
#include "symbol_manager.hpp"

af_err af_get_default_random_engine(af_random_engine *r) {
    CALL(af_get_default_random_engine, r);
}

af_err af_create_random_engine(af_random_engine *engineHandle,
                               af_random_engine_type rtype,
                               unsigned long long seed) {
    CALL(af_create_random_engine, engineHandle, rtype, seed);
}

af_err af_retain_random_engine(af_random_engine *outHandle,
                               const af_random_engine engineHandle) {
    CALL(af_retain_random_engine, outHandle, engineHandle);
}

af_err af_random_engine_get_type(af_random_engine_type *rtype,
                                 const af_random_engine engine) {
    CALL(af_random_engine_get_type, rtype, engine);
}

af_err af_random_engine_set_type(af_random_engine *engine,
                                 const af_random_engine_type rtype) {
    CALL(af_random_engine_set_type, engine, rtype);
}

af_err af_set_default_random_engine_type(const af_random_engine_type rtype) {
    CALL(af_set_default_random_engine_type, rtype);
}

af_err af_random_uniform(af_array *arr, const unsigned ndims,
                         const dim_t *const dims, const af_dtype type,
                         af_random_engine engine) {
    CALL(af_random_uniform, arr, ndims, dims, type, engine);
}

af_err af_random_normal(af_array *arr, const unsigned ndims,
                        const dim_t *const dims, const af_dtype type,
                        af_random_engine engine) {
    CALL(af_random_normal, arr, ndims, dims, type, engine);
}

af_err af_release_random_engine(af_random_engine engineHandle) {
    CALL(af_release_random_engine, engineHandle);
}

af_err af_random_engine_set_seed(af_random_engine *engine,
                                 const unsigned long long seed) {
    CALL(af_random_engine_set_seed, engine, seed);
}

af_err af_random_engine_get_seed(unsigned long long *const seed,
                                 af_random_engine engine) {
    CALL(af_random_engine_get_seed, seed, engine);
}

af_err af_randu(af_array *out, const unsigned ndims, const dim_t *const dims,
                const af_dtype type) {
    CALL(af_randu, out, ndims, dims, type);
}

af_err af_randn(af_array *out, const unsigned ndims, const dim_t *const dims,
                const af_dtype type) {
    CALL(af_randn, out, ndims, dims, type);
}

af_err af_set_seed(const unsigned long long seed) { CALL(af_set_seed, seed); }

af_err af_get_seed(unsigned long long *seed) { CALL(af_get_seed, seed); }
