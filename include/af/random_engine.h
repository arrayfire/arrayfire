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

            randomEngine& operator= (const randomEngine& other);

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

            void setType(const randomType type);
            void setSeed(const uintl seed);
            uintl getSeed(void) const;
            af_random_engine get() const;
    };

    /**
        \param[in] dims is the dimensions of the array to be generated
        \param[in] ty is the type of the array

        \return array of size \p dims

        \ingroup data_func_randu
    */
    AFAPI array randu(const dim4 &dims, const dtype ty=f32);

    /**
        \param[in] d0 is the size of the first dimension
        \param[in] ty is the type of the array

        \return array of size \p d0

        \ingroup data_func_randu
    */
    AFAPI array randu(const dim_t d0, const dtype ty=f32);

    /**
        \param[in] d0 is the size of the first dimension
        \param[in] d1 is the size of the second dimension
        \param[in] ty is the type of the array

        \return array of size \p d0 x \p d1

        \ingroup data_func_randu
    */
    AFAPI array randu(const dim_t d0,
                      const dim_t d1, const dtype ty=f32);

    /**
        \param[in] d0 is the size of the first dimension
        \param[in] d1 is the size of the second dimension
        \param[in] d2 is the size of the third dimension
        \param[in] ty is the type of the array

        \return array of size \p d0 x \p d1 x \p d2

        \ingroup data_func_randu
    */
    AFAPI array randu(const dim_t d0,
                      const dim_t d1, const dim_t d2, const dtype ty=f32);

    /**
        \param[in] d0 is the size of the first dimension
        \param[in] d1 is the size of the second dimension
        \param[in] d2 is the size of the third dimension
        \param[in] d3 is the size of the fourth dimension
        \param[in] ty is the type of the array

        \return array of size \p d0 x \p d1 x \p d2 x \p d3

        \ingroup data_func_randu
    */
    AFAPI array randu(const dim_t d0,
                      const dim_t d1, const dim_t d2,
                      const dim_t d3, const dtype ty=f32);

    /**
        \param[in] dims is the dimensions of the array to be generated
        \param[in] ty is the type of the array

        \return array of size \p dims

        \ingroup data_func_randn
    */
    AFAPI array randn(const dim4 &dims, const dtype ty=f32);

    /**
        \param[in] d0 is the size of the first dimension
        \param[in] ty is the type of the array

        \return array of size \p d0

        \ingroup data_func_randn
    */
    AFAPI array randn(const dim_t d0, const dtype ty=f32);
    /**
        \param[in] d0 is the size of the first dimension
        \param[in] d1 is the size of the second dimension
        \param[in] ty is the type of the array

        \return array of size \p d0 x \p d1

        \ingroup data_func_randn
    */
    AFAPI array randn(const dim_t d0,
                      const dim_t d1, const dtype ty=f32);
    /**
        \param[in] d0 is the size of the first dimension
        \param[in] d1 is the size of the second dimension
        \param[in] d2 is the size of the third dimension
        \param[in] ty is the type of the array

        \return array of size \p d0 x \p d1 x \p d2

        \ingroup data_func_randn
    */
    AFAPI array randn(const dim_t d0,
                      const dim_t d1, const dim_t d2, const dtype ty=f32);

    /**
        \param[in] d0 is the size of the first dimension
        \param[in] d1 is the size of the second dimension
        \param[in] d2 is the size of the third dimension
        \param[in] d3 is the size of the fourth dimension
        \param[in] ty is the type of the array

        \return array of size \p d0 x \p d1 x \p d2 x \p d3

        \ingroup data_func_randn
    */
    AFAPI array randn(const dim_t d0,
                      const dim_t d1, const dim_t d2,
                      const dim_t d3, const dtype ty=f32);

    /**
        \param[in] rtype is the type of the random number generator

        \ingroup data_func_set_type
    */
    AFAPI void setDefaultRandomEngine(randomType rtype);

    /**
        \param[in] seed is a 64 bit unsigned integer

        \ingroup data_func_setseed
    */
    AFAPI void setSeed(const uintl seed);

    /**
        \returns seed which is a 64 bit unsigned integer

        \ingroup data_func_getseed
    */
    AFAPI uintl getSeed();

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
       C Interface for retaining random engine

       \param[out]  out is the pointer to the returned random engine object
       \param[in]   engine is the random engine object

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_retain_random_engine(af_random_engine *out, const af_random_engine engine);

    /**
       C Interface for changing random engine type

       \param[in]   engine is the random engine object
       \param[in]   rtype is the type of the random number generator

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_random_engine_set_type(af_random_engine *engine, const af_random_type rtype);

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

       \param[out]  engine is the random engine object
       \param[in]   seed is the initializing seed of the random number generator

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_random_engine_set_seed(af_random_engine *engine, const uintl seed);

    /**
       C Interface for setting the type of the default random engine

       \param[in]   rtype is the type of the random number generator

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_default_random_engine_set_type(const af_random_type rtype);

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

    //General rand calls

    /**
        \param[out] out is the generated array
        \param[in] ndims is size of dimension array \p dims
        \param[in] dims is the array containing sizes of the dimension
        \param[in] type is the type of array to generate

       \ingroup data_func_randu
    */
    AFAPI af_err af_randu(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type);

    /**
        \param[out] out is the generated array
        \param[in] ndims is size of dimension array \p dims
        \param[in] dims is the array containing sizes of the dimension
        \param[in] type is the type of array to generate

       \ingroup data_func_randn
    */
    AFAPI af_err af_randn(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type);

    /**
        \param[in] seed is a 64 bit unsigned integer

        \ingroup data_func_setseed
    */
    AFAPI af_err af_set_seed(const uintl seed);

    /**
        \param[out] seed which is a 64 bit unsigned integer

        \ingroup data_func_getseed
    */
    AFAPI af_err af_get_seed(uintl *seed);

#ifdef __cplusplus
}
#endif
