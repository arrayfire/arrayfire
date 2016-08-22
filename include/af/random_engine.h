/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>

typedef void * af_random_engine;

#ifdef __cplusplus
namespace af
{
    class array;
    class dim4;

    class AFAPI randomEngine {
        private:
            af_random_engine engine;
        public:
            explicit
            randomEngine(randomType typeIn = AF_RANDOM_DEFAULT, uintl seedIn = 0);
            ~randomEngine();

            array uniform(const dim_t dim0, const dtype ty = f32);
            array uniform(const dim_t dim0, const dim_t dim1, const dtype ty = f32);
            array uniform(const dim_t dim0, const dim_t dim1, const dim_t dim2, const dtype ty = f32);
            array uniform(const dim_t dim0, const dim_t dim1, const dim_t dim2, const dim_t dim3, const dtype ty = f32);
            array uniform(const dim4& dims, const dtype ty = f32);

            array normal(const dim_t dim0, const dtype ty = f32);
            array normal(const dim_t dim0, const dim_t dim1, const dtype ty = f32);
            array normal(const dim_t dim0, const dim_t dim1, const dim_t dim2, const dtype ty = f32);
            array normal(const dim_t dim0, const dim_t dim1, const dim_t dim2, const dim_t dim3, const dtype ty = f32);
            array normal(const dim4& dims, const dtype ty = f32);

            void setSeed(uintl seed);
            uintl getSeed();
    };
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       C Interface for creating random engine

       \param[out]  engine is the pointer to the returned random engine object
       \param[in]   rtype is the type of the random number generator
       \param[in]   seed is the initializing seed of the random number generator

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_create_random_engine(af_random_engine *engine, af_random_type rtype, uintl seed);

    /**
       C Interface for creating an array of uniform numbers using a random engine

       \param[out]  out The pointer to the returned object.
       \param[in]   engine is the random engine object
       \param[in]   ndims The number of dimensions read from the \p dims parameter
       \param[in]   dims A C pointer with \p ndims elements. Each value represents the size of that dimension
       \param[in]   type The type of the \ref af_array object

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_random_engine_uniform(af_array *out, af_random_engine engine, const unsigned ndims, const dim_t * const dims, const af_dtype type);

    /**
       C Interface for creating an array of normal numbers using a random engine

       \param[out]  out The pointer to the returned object.
       \param[in]   engine is the random engine object
       \param[in]   ndims The number of dimensions read from the \p dims parameter
       \param[in]   dims A C pointer with \p ndims elements. Each value represents the size of that dimension
       \param[in]   type The type of the \ref af_array object

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_random_engine_normal(af_array *out, af_random_engine engine, const unsigned ndims, const dim_t * const dims, const af_dtype type);

    /**
       C Interface for setting the seed of a random engine

       \param[in]   engine is the random engine object
       \param[in]   seed is the initializing seed of the random number generator

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_random_engine_set_seed(const uintl seed, af_random_engine engine);

    /**
       C Interface for getting the seed of a random engine

       \param[out]  out The pointer to the returned seed.
       \param[in]   engine is the random engine object

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_random_engine_get_seed(uintl * const seed, af_random_engine engine);

    /**
       C Interface for releasing random engine

       \param[in] engine is the random engine object
       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_release_random_engine(af_random_engine engine);

#ifdef __cplusplus
}
#endif
