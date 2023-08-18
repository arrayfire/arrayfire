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

///
/// \brief Handle for a random engine object.
///
/// This handle is used to reference the internal random engine object.
///
/// \ingroup random_mat
typedef void * af_random_engine;

#ifdef __cplusplus
namespace af
{
    class array;
    class dim4;
#if AF_API_VERSION >= 34
    /// C++ Interface - Random Number Generation Engine Class
    ///
    /// The \ref af::randomEngine class is used to set the type and seed of
    /// random number generation engine based on \ref af::randomEngineType.
    ///
    /// \ingroup arrayfire_class
    /// \ingroup random_mat
    class AFAPI randomEngine {
    private:
      ///
      /// \brief Handle to the interal random engine object
      af_random_engine engine;

    public:
      /**
          C++ Interface to create a \ref af::randomEngine object with a \ref
          af::randomEngineType and a seed.

          \code
            // create a random engine of default type with seed = 1
            randomEngine r(AF_RANDOM_ENGINE_DEFAULT, 1);
          \endcode
      */
      explicit randomEngine(randomEngineType typeIn = AF_RANDOM_ENGINE_DEFAULT,
                            unsigned long long seedIn = 0);

      /**
          C++ Interface copy constructor for a \ref af::randomEngine.

          \param[in] other input random engine object
      */
      randomEngine(const randomEngine &other);

      /**
          C++ Interface to create a copy of the random engine object from a
          \ref af_random_engine handle.

          \param[in] engine The input random engine object
      */
      randomEngine(af_random_engine engine);

      /**
          C++ Interface destructor for a \ref af::randomEngine.
      */
      ~randomEngine();

      /**
          C++ Interface to assign the internal state of randome engine.

          \param[in] other object to be assigned to the random engine

          \return the reference to this
      */
      randomEngine &operator=(const randomEngine &other);

      /**
          C++ Interface to set the random type of the random engine.

          \param[in] type type of the random number generator
      */
      void setType(const randomEngineType type);

      /**
          C++ Interface to get the random type of the random engine.

          \return \ref af::randomEngineType associated with random engine
      */
      randomEngineType getType(void);

      /**
          C++ Interface to set the seed of the random engine.

          \param[in] seed initializing seed of the random number generator
      */
      void setSeed(const unsigned long long seed);

      /**
          C++ Interface to return the seed of the random engine.

          \return seed associated with random engine
      */
      unsigned long long getSeed(void) const;

      /**
          C++ Interface to return the af_random_engine handle of this object.

          \return handle to the af_random_engine associated with this random
                  engine
      */
      af_random_engine get(void) const;
    };
#endif

#if AF_API_VERSION >= 34
    /**
        C++ Interface to create an array of random numbers uniformly
        distributed.

        \param[in] dims dimensions of the array to be generated
        \param[in] ty   type of the array
        \param[in] r    random engine object
        \return    random number array of size `dims`

        \ingroup random_func_randu
    */
    AFAPI array randu(const dim4 &dims, const dtype ty, randomEngine &r);
#endif

#if AF_API_VERSION >= 34
    /**
        C++ Interface to create an array of random numbers normally
        distributed.

        \param[in] dims dimensions of the array to be generated
        \param[in] ty   type of the array
        \param[in] r    random engine object
        \return    random number array of size `dims`

        \ingroup random_func_randn
    */
    AFAPI array randn(const dim4 &dims, const dtype ty, randomEngine &r);
#endif

    /**
        C++ Interface to create an array of random numbers uniformly
        distributed.

        \param[in] dims dimensions of the array to be generated
        \param[in] ty   type of the array

        \ingroup random_func_randu
    */
    AFAPI array randu(const dim4 &dims, const dtype ty=f32);

    /**
        C++ Interface to create an array of random numbers uniformly
        distributed.

        \param[in] d0 size of the first dimension
        \param[in] ty type of the array
        \return    random number array of size `d0`

        \ingroup random_func_randu
    */
    AFAPI array randu(const dim_t d0, const dtype ty=f32);

    /**
        C++ Interface to create an array of random numbers uniformly
        distributed.

        \param[in] d0 size of the first dimension
        \param[in] d1 size of the second dimension
        \param[in] ty type of the array
        \return    random number array of size `d0` x `d1`

        \ingroup random_func_randu
    */
    AFAPI array randu(const dim_t d0,
                      const dim_t d1, const dtype ty=f32);

    /**
        C++ Interface to create an array of random numbers uniformly
        distributed.

        \param[in] d0 size of the first dimension
        \param[in] d1 size of the second dimension
        \param[in] d2 size of the third dimension
        \param[in] ty type of the array
        \return    random number array of size `d0` x `d1` x `d2`

        \ingroup random_func_randu
    */
    AFAPI array randu(const dim_t d0,
                      const dim_t d1, const dim_t d2, const dtype ty=f32);

    /**
        C++ Interface to create an array of random numbers uniformly
        distributed.

        \param[in] d0 size of the first dimension
        \param[in] d1 size of the second dimension
        \param[in] d2 size of the third dimension
        \param[in] d3 size of the fourth dimension
        \param[in] ty type of the array
        \return    random number array of size `d0` x `d1` x `d2` x `d3`

        \ingroup random_func_randu
    */
    AFAPI array randu(const dim_t d0,
                      const dim_t d1, const dim_t d2,
                      const dim_t d3, const dtype ty=f32);

    /**
        C++ Interface to create an array of random numbers normally
        distributed.

        \param[in] dims dimensions of the array to be generated
        \param[in] ty   type of the array
        \return    random number array of size `dims`

        \ingroup random_func_randn
    */
    AFAPI array randn(const dim4 &dims, const dtype ty=f32);

    /**
        C++ Interface to create an array of random numbers normally
        distributed.

        \param[in] d0 size of the first dimension
        \param[in] ty type of the array
        \return    random number array of size `d0`

        \ingroup random_func_randn
    */
    AFAPI array randn(const dim_t d0, const dtype ty=f32);
    /**
        C++ Interface to create an array of random numbers normally
        distributed.

        \param[in] d0 size of the first dimension
        \param[in] d1 size of the second dimension
        \param[in] ty type of the array
        \return    random number array of size `d0` x `d1`

        \ingroup random_func_randn
    */
    AFAPI array randn(const dim_t d0,
                      const dim_t d1, const dtype ty=f32);
    /**
        C++ Interface to create an array of random numbers normally
        distributed.

        \param[in] d0 size of the first dimension
        \param[in] d1 size of the second dimension
        \param[in] d2 size of the third dimension
        \param[in] ty type of the array
        \return    random number array of size `d0` x `d1` x `d2`

        \ingroup random_func_randn
    */
    AFAPI array randn(const dim_t d0,
                      const dim_t d1, const dim_t d2, const dtype ty=f32);

    /**
        C++ Interface to create an array of random numbers normally
        distributed.

        \param[in] d0 size of the first dimension
        \param[in] d1 size of the second dimension
        \param[in] d2 size of the third dimension
        \param[in] d3 size of the fourth dimension
        \param[in] ty type of the array
        \return    random number array of size `d0` x `d1` x `d2` x `d3`

        \ingroup random_func_randn
    */
    AFAPI array randn(const dim_t d0,
                      const dim_t d1, const dim_t d2,
                      const dim_t d3, const dtype ty=f32);

#if AF_API_VERSION >= 34
    /**
        C++ Interface to set the default random engine type.

        \param[in] rtype type of the random number generator

        \ingroup random_func_set_default_engine
    */
    AFAPI void setDefaultRandomEngineType(randomEngineType rtype);
#endif

#if AF_API_VERSION >= 34
    /**
        C++ Interface to get the default random engine type.

        \return \ref af::randomEngine object for the default random engine

        \ingroup random_func_get_default_engine
    */
    AFAPI randomEngine getDefaultRandomEngine(void);
#endif

    /**
        C++ Interface to set the seed of the default random number generator.

        \param[in] seed 64-bit unsigned integer

        \ingroup random_func_set_seed
    */
    AFAPI void setSeed(const unsigned long long seed);

    /**
        C++ Interface to get the seed of the default random number generator.

        \return seed 64-bit unsigned integer

        \ingroup random_func_get_seed
    */
    AFAPI unsigned long long getSeed();

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface to create a random engine.

       \param[out] engine pointer to the returned random engine object
       \param[in]  rtype  type of the random number generator
       \param[in]  seed   initializing seed of the random number generator
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup random_func_random_engine
    */
    AFAPI af_err af_create_random_engine(af_random_engine *engine,
                                         af_random_engine_type rtype,
                                         unsigned long long seed);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface to retain a random engine.

       \param[out] out    pointer to the returned random engine object
       \param[in]  engine random engine object
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup random_func_random_engine
    */
    AFAPI af_err af_retain_random_engine(af_random_engine *out,
                                         const af_random_engine engine);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface to change random engine type.

       \param[in]  engine random engine object
       \param[in]  rtype  type of the random number generator
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup random_func_random_engine
    */
    AFAPI af_err af_random_engine_set_type(af_random_engine *engine,
                                           const af_random_engine_type rtype);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface to get random engine type.

       \param[out] rtype  type of the random number generator
       \param[in]  engine random engine object
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup random_func_random_engine
    */
    AFAPI af_err af_random_engine_get_type(af_random_engine_type *rtype,
                                           const af_random_engine engine);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface to create an array of uniform numbers using a random engine.

       \param[out] out    pointer to the returned object
       \param[in]  ndims  number of dimensions
       \param[in]  dims   C pointer with `ndims` elements; each value
                          represents the size of that dimension
       \param[in]  type   type of the \ref af_array object
       \param[in]  engine random engine object
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup random_func_randu
    */
    AFAPI af_err af_random_uniform(af_array *out, const unsigned ndims,
                                   const dim_t * const dims, const af_dtype type,
                                   af_random_engine engine);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface to create an array of normal numbers using a random engine.

       \param[out] out    pointer to the returned object
       \param[in]  ndims  number of dimensions
       \param[in]  dims   C pointer with `ndims` elements; each value
                          represents the size of that dimension
       \param[in]  type   type of the \ref af_array object
       \param[in]  engine random engine object
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup random_func_randn
    */
    AFAPI af_err af_random_normal(af_array *out, const unsigned ndims,
                                  const dim_t * const dims, const af_dtype type,
                                  af_random_engine engine);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface to set the seed of a random engine.

       \param[out] engine pointer to the returned random engine object
       \param[in]  seed   initializing seed of the random number generator
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup random_func_random_engine
    */
    AFAPI af_err af_random_engine_set_seed(af_random_engine *engine,
                                           const unsigned long long seed);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface to get the default random engine.

       \param[out] engine pointer to the returned default random engine object
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup random_func_get_default_engine
    */
    AFAPI af_err af_get_default_random_engine(af_random_engine *engine);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface to set the type of the default random engine.

       \param[in]  rtype type of the random number generator
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup random_func_set_default_engine
    */
    AFAPI af_err af_set_default_random_engine_type(const af_random_engine_type rtype);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface to get the seed of a random engine.

       \param[out] seed   pointer to the returned seed
       \param[in]  engine random engine object
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup random_func_random_engine
    */
    AFAPI af_err af_random_engine_get_seed(unsigned long long * const seed,
                                           af_random_engine engine);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface to release a random engine.

       \param[in] engine random engine object
       \return    \ref AF_SUCCESS, if function returns successfully, else
                  an \ref af_err code is given

       \ingroup random_func_random_engine
    */
    AFAPI af_err af_release_random_engine(af_random_engine engine);
#endif

    /**
       \param[out] out   generated array
       \param[in]  ndims number of dimensions
       \param[in]  dims  array containing sizes of the dimension
       \param[in]  type  type of array to generate
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup random_func_randu
    */
    AFAPI af_err af_randu(af_array *out, const unsigned ndims,
                          const dim_t * const dims, const af_dtype type);

    /**
       \param[out] out   generated array
       \param[in]  ndims number of dimensions
       \param[in]  dims  array containing sizes of the dimension
       \param[in]  type  type of array to generate
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

       \ingroup random_func_randn
    */
    AFAPI af_err af_randn(af_array *out, const unsigned ndims,
                          const dim_t * const dims, const af_dtype type);

    /**
       \param[in] seed a 64-bit unsigned integer
       \return    \ref AF_SUCCESS, if function returns successfully, else
                  an \ref af_err code is given

        \ingroup random_func_set_seed
    */
    AFAPI af_err af_set_seed(const unsigned long long seed);

    /**
       \param[out] seed a 64-bit unsigned integer
       \return     \ref AF_SUCCESS, if function returns successfully, else
                   an \ref af_err code is given

        \ingroup random_func_get_seed
    */
    AFAPI af_err af_get_seed(unsigned long long *seed);

#ifdef __cplusplus
}
#endif
