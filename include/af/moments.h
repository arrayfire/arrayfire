/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>

#ifdef __cplusplus
namespace af
{
class array;

#if AF_API_VERSION >= 34
/**
   C++ Interface for calculating an image moment

   \param[in]  in is the input image
   \param[moment] is the moment to calculate
   \return      the value of the moment

   \ingroup image_func_moments
 */
template<typename T> T moment(const array& in, const af_moment_type moment);
#endif

#if AF_API_VERSION >= 34
/**
   C++ Interface for calculating image moments

   \param[in]  in contains the input image(s)
   \param[moment] is the moment to calculate
   \return array containing the requested moment of each image

   \ingroup image_func_moments
 */
AFAPI array moments(const array& in, const af_moment_type moment);
#endif

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface for finding image moments

       \param[out] out is an array containing the calculated moments
       \param[in]  in is an array of image(s)
       \param[moment] is the moment to calculate
       \return     ref AF_SUCCESS if the moment calculation is successful,
       otherwise an appropriate error code is returned.

       \ingroup image_func_moments
    */
    AFAPI af_err af_moments(af_array *out, const af_array in, const af_moment_type moment);
#endif

#if AF_API_VERSION >= 34
    /**
       C Interface for calculating an image moment

       \param[out] out is a pointer to the outputted moment
       \param[in]  in is an array of image(s)
       \param[moment] is the moment to calculate
       \return     ref AF_SUCCESS if the moment calculation is successful,
       otherwise an appropriate error code is returned.

       \ingroup image_func_moments
    */
    AFAPI af_err af_moment(double *out, const af_array in, const af_moment_type moment);
#endif

#ifdef __cplusplus
}
#endif
