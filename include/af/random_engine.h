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

    ///
    /// \brief A random number generator class
    ///
    class AFAPI randomEngine {
        private:
            af_random_engine engine;
        public:
            /**
                \ingroup construct_random
                @{
            */
            /**
                Create random number generator object

                \code
                randomEngine r(AF_RANDOM_ENGINE_DEFAULT, 1);   // creates random engine of default type with seed = 1
                \endcode
            */
            explicit
            randomEngine(randomEngineType typeIn = AF_RANDOM_ENGINE_DEFAULT, uintl seedIn = 0);
            /**
                Creates a copy of the random engine object
                \param in The input random engine object
            */
            randomEngine(const randomEngine& in);
            /**
                @}
            */

            ~randomEngine();

            /**
                \ingroup random_operator_eq
                @{
                \brief Assigns the internal state of randome engine

                \param[in] in The object to be assigned to the random engine
                \returns the reference to this

            */
            randomEngine& operator= (const randomEngine& in);
            /**
                @}
            */

            /**
                \ingroup random_set_type
                @{
                \brief Sets the random type of the random engine

                \param[in] type The type of the random number generator
            */
            void setType(const randomEngineType type);
            /**
                @}
            */

            /**
                \ingroup random_get_type
                @{
                \brief Return the random type of the random engine

                \returns the random type enum associated with random engine
            */
            randomEngineType getType(void);
            /**
                @}
            */

            /**
                \ingroup random_set_seed
                @{
                \brief Sets the seed of the random engine

                \param[in] seed The initializing seed of the random number generator
            */
            void setSeed(const uintl seed);
            /**
                @}
            */

            /**
                \ingroup random_get_seed
                @{
                \brief Returns the seed of the random engine

                \returns the seed associated with random engine
            */
            uintl getSeed(void) const;
            /**
                @}
            */

            /**
                \ingroup random_get
                @{
                \brief Returns the internal state of the random engine

                \returns the internal state associated with random engine
            */
            af_random_engine get(void) const;
            /**
                @}
            */
    };

    /**
        \param[in] dims The dimensions of the array to be generated
        \param[in] ty The type of the array
        \param[in] r The random engine object

        \return array of size \p dims

        \ingroup random_func_randu
    */
    AFAPI array randu(const dim4 &dims, const dtype ty, randomEngine &r);

    /**
        \param[in] dims The dimensions of the array to be generated
        \param[in] ty The type of the array
        \param[in] r The random engine object

        \return array of size \p dims

        \ingroup random_func_randn
    */
    AFAPI array randn(const dim4 &dims, const dtype ty, randomEngine &r);

    /**
        \param[in] dims The dimensions of the array to be generated
        \param[in] ty The type of the array

        \return array of size \p dims

        \ingroup random_func_randu
    */
    AFAPI array randu(const dim4 &dims, const dtype ty=f32);

    /**
        \param[in] d0 The size of the first dimension
        \param[in] ty The type of the array

        \return array of size \p d0

        \ingroup random_func_randu
    */
    AFAPI array randu(const dim_t d0, const dtype ty=f32);

    /**
        \param[in] d0 The size of the first dimension
        \param[in] d1 The size of the second dimension
        \param[in] ty The type of the array

        \return array of size \p d0 x \p d1

        \ingroup random_func_randu
    */
    AFAPI array randu(const dim_t d0,
                      const dim_t d1, const dtype ty=f32);

    /**
        \param[in] d0 The size of the first dimension
        \param[in] d1 The size of the second dimension
        \param[in] d2 The size of the third dimension
        \param[in] ty The type of the array

        \return array of size \p d0 x \p d1 x \p d2

        \ingroup random_func_randu
    */
    AFAPI array randu(const dim_t d0,
                      const dim_t d1, const dim_t d2, const dtype ty=f32);

    /**
        \param[in] d0 The size of the first dimension
        \param[in] d1 The size of the second dimension
        \param[in] d2 The size of the third dimension
        \param[in] d3 The size of the fourth dimension
        \param[in] ty The type of the array

        \return array of size \p d0 x \p d1 x \p d2 x \p d3

        \ingroup random_func_randu
    */
    AFAPI array randu(const dim_t d0,
                      const dim_t d1, const dim_t d2,
                      const dim_t d3, const dtype ty=f32);

    /**
        \param[in] dims The dimensions of the array to be generated
        \param[in] ty The type of the array

        \return array of size \p dims

        \ingroup random_func_randn
    */
    AFAPI array randn(const dim4 &dims, const dtype ty=f32);

    /**
        \param[in] d0 The size of the first dimension
        \param[in] ty The type of the array

        \return array of size \p d0

        \ingroup random_func_randn
    */
    AFAPI array randn(const dim_t d0, const dtype ty=f32);
    /**
        \param[in] d0 The size of the first dimension
        \param[in] d1 The size of the second dimension
        \param[in] ty The type of the array

        \return array of size \p d0 x \p d1

        \ingroup random_func_randn
    */
    AFAPI array randn(const dim_t d0,
                      const dim_t d1, const dtype ty=f32);
    /**
        \param[in] d0 The size of the first dimension
        \param[in] d1 The size of the second dimension
        \param[in] d2 The size of the third dimension
        \param[in] ty The type of the array

        \return array of size \p d0 x \p d1 x \p d2

        \ingroup random_func_randn
    */
    AFAPI array randn(const dim_t d0,
                      const dim_t d1, const dim_t d2, const dtype ty=f32);

    /**
        \param[in] d0 The size of the first dimension
        \param[in] d1 The size of the second dimension
        \param[in] d2 The size of the third dimension
        \param[in] d3 The size of the fourth dimension
        \param[in] ty The type of the array

        \return array of size \p d0 x \p d1 x \p d2 x \p d3

        \ingroup random_func_randn
    */
    AFAPI array randn(const dim_t d0,
                      const dim_t d1, const dim_t d2,
                      const dim_t d3, const dtype ty=f32);

    /**
        \param[in] rtype The type of the random number generator

        \ingroup random_func_set_type
    */
    AFAPI void setDefaultRandomEngine(randomEngineType rtype);

    /**
        \param[in] seed A 64 bit unsigned integer

        \ingroup random_func_setseed
    */
    AFAPI void setSeed(const uintl seed);

    /**
        \returns seed A 64 bit unsigned integer

        \ingroup random_func_getseed
    */
    AFAPI uintl getSeed();

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
       C Interface for creating random engine

       \param[out]  engine The pointer to the returned random engine object
       \param[in]   rtype The type of the random number generator
       \param[in]   seed The initializing seed of the random number generator

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_create_random_engine(af_random_engine *engine, af_random_engine_type rtype, uintl seed);

    /**
       C Interface for retaining random engine

       \param[out]  out The pointer to the returned random engine object
       \param[in]   engine The random engine object

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_retain_random_engine(af_random_engine *out, const af_random_engine engine);

    /**
       C Interface for changing random engine type

       \param[in]   engine The random engine object
       \param[in]   rtype The type of the random number generator

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_random_engine_set_type(af_random_engine *engine, const af_random_engine_type rtype);

    /**
       C Interface for getting random engine type

       \param[out]  rtype The type of the random number generator
       \param[in]   engine The random engine object

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_random_engine_get_type(af_random_engine_type *rtype, const af_random_engine engine);

    /**
       C Interface for creating an array of uniform numbers using a random engine

       \param[out]  out The pointer to the returned object.
       \param[in]   ndims The number of dimensions read from the \p dims parameter
       \param[in]   dims A C pointer with \p ndims elements. Each value represents the size of that dimension
       \param[in]   type The type of the \ref af_array object
       \param[in]   engine The random engine object

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_random_uniform(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type, af_random_engine engine);

    /**
       C Interface for creating an array of normal numbers using a random engine

       \param[out]  out The pointer to the returned object.
       \param[in]   ndims The number of dimensions read from the \p dims parameter
       \param[in]   dims A C pointer with \p ndims elements. Each value represents the size of that dimension
       \param[in]   type The type of the \ref af_array object
       \param[in]   engine The random engine object

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_random_normal(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type, af_random_engine engine);

    /**
       C Interface for setting the seed of a random engine

       \param[out]  engine The pointer to the returned random engine object
       \param[in]   seed The initializing seed of the random number generator

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_random_engine_set_seed(af_random_engine *engine, const uintl seed);

    /**
       C Interface for getting the default random engine

       \param[out]  engine The pointer to returned default random engine object

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_get_default_random_engine(af_random_engine *engine);

    /**
       C Interface for setting the type of the default random engine

       \param[in]   rtype The type of the random number generator

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_set_default_random_engine(const af_random_engine_type rtype);

    /**
       C Interface for getting the seed of a random engine

       \param[out]  seed The pointer to the returned seed.
       \param[in]   engine The random engine object

       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_random_engine_get_seed(uintl * const seed, af_random_engine engine);

    /**
       C Interface for releasing random engine

       \param[in] engine The random engine object
       \returns \ref AF_SUCCESS if the execution completes properly
    */
    AFAPI af_err af_release_random_engine(af_random_engine engine);

    //General rand calls

    /**
        \param[out] out The generated array
        \param[in] ndims Size of dimension array \p dims
        \param[in] dims The array containing sizes of the dimension
        \param[in] type The type of array to generate

       \ingroup random_func_randu
    */
    AFAPI af_err af_randu(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type);

    /**
        \param[out] out The generated array
        \param[in] ndims Size of dimension array \p dims
        \param[in] dims The array containing sizes of the dimension
        \param[in] type The type of array to generate

       \ingroup random_func_randn
    */
    AFAPI af_err af_randn(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type);

    /**
        \param[in] seed A 64 bit unsigned integer

        \ingroup random_func_setseed
    */
    AFAPI af_err af_set_seed(const uintl seed);

    /**
        \param[out] seed A 64 bit unsigned integer

        \ingroup random_func_getseed
    */
    AFAPI af_err af_get_seed(uintl *seed);

#ifdef __cplusplus
}
#endif
