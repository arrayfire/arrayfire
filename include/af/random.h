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
/// \brief Handle for random engine
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
    /// \brief Random Number Generation Engine Class
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
          This function creates a \ref af::randomEngine object with a
          \ref af::randomEngineType and a seed.

          \code
          // creates random engine of default type with seed = 1
          randomEngine r(AF_RANDOM_ENGINE_DEFAULT, 1);
         \endcode
      */
      explicit randomEngine(randomEngineType typeIn = AF_RANDOM_ENGINE_DEFAULT,
                            unsigned long long seedIn = 0);

      /**
          Copy constructor for \ref af::randomEngine.

          \param[in] other The input random engine object
      */
      randomEngine(const randomEngine &other);

      /**
          Creates a copy of the random engine object from a \ref
          af_random_engine handle.

          \param[in] engine The input random engine object
      */
      randomEngine(af_random_engine engine);

      /**
          \brief Destructor for \ref af::randomEngine
      */
      ~randomEngine();

      /**
          \brief Assigns the internal state of randome engine

          \param[in] other The object to be assigned to the random engine

          \returns the reference to this
      */
      randomEngine &operator=(const randomEngine &other);

      /**
          \brief Sets the random type of the random engine

          \param[in] type The type of the random number generator
      */
      void setType(const randomEngineType type);

      /**
          \brief Return the random type of the random engine

          \returns the \ref af::randomEngineType associated with random engine
      */
      randomEngineType getType(void);

      /**
          \brief Sets the seed of the random engine

          \param[in] seed The initializing seed of the random number generator
      */
      void setSeed(const unsigned long long seed);

      /**
          \brief Returns the seed of the random engine

          \returns the seed associated with random engine
      */
      unsigned long long getSeed(void) const;

      /**
          \brief Returns the af_random_engine handle of this object

          \returns the handle to the af_random_engine associated with this
                   random engine
      */
      af_random_engine get(void) const;
    };
#endif

#if AF_API_VERSION >= 34
    /**
        \param[in] dims The dimensions of the array to be generated
        \param[in] ty The type of the array
        \param[in] r The random engine object

        \return array of size \p dims

        \ingroup random_func_randu
    */
    AFAPI array randu(const dim4 &dims, const dtype ty, randomEngine &r);
#endif

#if AF_API_VERSION >= 34
    /**
        \param[in] dims The dimensions of the array to be generated
        \param[in] ty The type of the array
        \param[in] r The random engine object

        \return array of size \p dims

        \ingroup random_func_randn
    */
    AFAPI array randn(const dim4 &dims, const dtype ty, randomEngine &r);
#endif

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

#if AF_API_VERSION >= 34
    /**
        \param[in] rtype The type of the random number generator

        \ingroup random_func_set_default_engine
    */
    AFAPI void setDefaultRandomEngineType(randomEngineType rtype);
#endif

#if AF_API_VERSION >= 34
    /**
        \returns the \ref af::randomEngine object for the default random engine

        \ingroup random_func_get_default_engine
    */
    AFAPI randomEngine getDefaultRandomEngine(void);
#endif

    /**
        \brief Sets the seed of the default random number generator

        \param[in] seed A 64 bit unsigned integer
        \ingroup random_func_set_seed
    */
    AFAPI void setSeed(const unsigned long long seed);

    /**
        \brief Gets the seed of the default random number generator

        \returns seed A 64 bit unsigned integer
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
       C Interface for creating random engine

       \param[out]  engine The pointer to the returned random engine object
       \param[in]   rtype The type of the random number generator
       \param[in]   seed The initializing seed of the random number generator

       \returns \ref AF_SUCCESS if the execution completes properly

       \ingroup random_func_random_engine
    */
    AFAPI af_err af_create_random_engine(af_random_engine *engine,
                                         af_random_engine_type rtype,
                                         unsigned long long seed);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface for retaining random engine

       \param[out]  out The pointer to the returned random engine object
       \param[in]   engine The random engine object

       \returns \ref AF_SUCCESS if the execution completes properly

       \ingroup random_func_random_engine
    */
    AFAPI af_err af_retain_random_engine(af_random_engine *out,
                                         const af_random_engine engine);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface for changing random engine type

       \param[in]   engine The random engine object
       \param[in]   rtype The type of the random number generator

       \returns \ref AF_SUCCESS if the execution completes properly

       \ingroup random_func_random_engine
    */
    AFAPI af_err af_random_engine_set_type(af_random_engine *engine,
                                           const af_random_engine_type rtype);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface for getting random engine type

       \param[out]  rtype The type of the random number generator
       \param[in]   engine The random engine object

       \returns \ref AF_SUCCESS if the execution completes properly

       \ingroup random_func_random_engine
    */
    AFAPI af_err af_random_engine_get_type(af_random_engine_type *rtype,
                                           const af_random_engine engine);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface for creating an array of uniform numbers using a random
       engine

       \param[out]  out The pointer to the returned object.
       \param[in]   ndims The number of dimensions read from the \p dims
                    parameter
       \param[in]   dims A C pointer with \p ndims elements. Each value
                    represents the size of that dimension
       \param[in]   type The type of the \ref af_array object
       \param[in]   engine The random engine object

       \returns \ref AF_SUCCESS if the execution completes properly

       \ingroup random_func_randu
    */
    AFAPI af_err af_random_uniform(af_array *out, const unsigned ndims,
                                   const dim_t * const dims, const af_dtype type,
                                   af_random_engine engine);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface for creating an array of normal numbers using a random engine

       \param[out]  out The pointer to the returned object.
       \param[in]   ndims The number of dimensions read from the \p dims
                    parameter
       \param[in]   dims A C pointer with \p ndims elements. Each value
                    represents the size of that dimension
       \param[in]   type The type of the \ref af_array object
       \param[in]   engine The random engine object

       \returns \ref AF_SUCCESS if the execution completes properly

       \ingroup random_func_randn
    */
    AFAPI af_err af_random_normal(af_array *out, const unsigned ndims,
                                  const dim_t * const dims, const af_dtype type,
                                  af_random_engine engine);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface for setting the seed of a random engine

       \param[out]  engine The pointer to the returned random engine object
       \param[in]   seed The initializing seed of the random number generator

       \returns \ref AF_SUCCESS if the execution completes properly

       \ingroup random_func_random_engine
    */
    AFAPI af_err af_random_engine_set_seed(af_random_engine *engine,
                                           const unsigned long long seed);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface for getting the default random engine

       \param[out]  engine The pointer to returned default random engine object

       \returns \ref AF_SUCCESS if the execution completes properly

       \ingroup random_func_get_default_engine
    */
    AFAPI af_err af_get_default_random_engine(af_random_engine *engine);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface for setting the type of the default random engine

       \param[in]   rtype The type of the random number generator

       \returns \ref AF_SUCCESS if the execution completes properly

       \ingroup random_func_set_default_engine
    */
    AFAPI af_err af_set_default_random_engine_type(const af_random_engine_type rtype);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface for getting the seed of a random engine

       \param[out]  seed The pointer to the returned seed.
       \param[in]   engine The random engine object

       \returns \ref AF_SUCCESS if the execution completes properly

       \ingroup random_func_random_engine
    */
    AFAPI af_err af_random_engine_get_seed(unsigned long long * const seed,
                                           af_random_engine engine);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface for releasing random engine

       \param[in] engine The random engine object
       \returns \ref AF_SUCCESS if the execution completes properly

       \ingroup random_func_random_engine
    */
    AFAPI af_err af_release_random_engine(af_random_engine engine);
#endif

    /**
        \param[out] out The generated array
        \param[in] ndims Size of dimension array \p dims
        \param[in] dims The array containing sizes of the dimension
        \param[in] type The type of array to generate

       \ingroup random_func_randu
    */
    AFAPI af_err af_randu(af_array *out, const unsigned ndims,
                          const dim_t * const dims, const af_dtype type);

    /**
        \param[out] out The generated array
        \param[in] ndims Size of dimension array \p dims
        \param[in] dims The array containing sizes of the dimension
        \param[in] type The type of array to generate

       \ingroup random_func_randn
    */
    AFAPI af_err af_randn(af_array *out, const unsigned ndims,
                          const dim_t * const dims, const af_dtype type);

    /**
        \param[in] seed A 64 bit unsigned integer

        \ingroup random_func_set_seed
    */
    AFAPI af_err af_set_seed(const unsigned long long seed);

    /**
        \param[out] seed A 64 bit unsigned integer

        \ingroup random_func_get_seed
    */
    AFAPI af_err af_get_seed(unsigned long long *seed);

#ifdef __cplusplus
}
#endif
