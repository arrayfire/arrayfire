/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/features.h>

#ifdef __cplusplus
namespace af
{
class array;

/**
    C++ Interface for FAST feature detector

    \param[in] in array containing a grayscale image (color images are not
               supported)
    \param[in] thr FAST threshold for which a pixel of the circle around
               the central pixel is considered to be greater or smaller
    \param[in] arc_length length of arc (or sequential segment) to be tested,
               must be within range [9-16]
    \param[in] non_max performs non-maximal suppression if true
    \param[in] feature_ratio maximum ratio of features to detect, the maximum
               number of features is calculated by feature_ratio * in.elements().
               The maximum number of features is not based on the score, instead,
               features detected after the limit is reached are discarded
    \param[in] edge is the length of the edges in the image to be discarded
               by FAST (minimum is 3, as the radius of the circle)
    \return    features object containing arrays for x and y coordinates and
               score, while array orientation is set to 0 as FAST does not
               compute orientation, and size is set to 1 as FAST does not
               compute multiple scales

    \ingroup cv_func_fast
 */
AFAPI features fast(const array& in, const float thr=20.0f, const unsigned arc_length=9, const bool non_max=true, const float feature_ratio=0.05, const unsigned edge=3);

/**
    C++ Interface for ORB feature descriptor

    \param[out] feat features object composed of arrays for x and y
                coordinates, score, orientation and size of selected features
    \param[out] desc Nx8 array containing extracted descriptors, where N is the
                number of selected features
    \param[in]  image array containing a grayscale image (color images are not
                supported)
    \param[in]  fast_thr FAST threshold for which a pixel of the circle around
                the central pixel is considered to be brighter or darker
    \param[in]  max_feat maximum number of features to hold (will only keep the
                max_feat features with higher Harris responses)
    \param[in]  scl_fctr factor to downsample the input image, meaning that
                each level will hold prior level dimensions divided by scl_fctr
    \param[in]  levels number of levels to be computed for the image pyramid
    \param[in]  blur_img blur image with a Gaussian filter with sigma=2 before
                computing descriptors to increase robustness against noise if
                true

    \ingroup cv_func_orb
 */
AFAPI void orb(features& feat, array& desc, const array& image, const float fast_thr=20.f, const unsigned max_feat=400, const float scl_fctr=1.5f, const unsigned levels=4, const bool blur_img=false);


/**
   C++ Interface wrapper for Hamming matcher

   \param[out] idx is an array of MxN size, where M is equal to the number of query
               features and N is equal to n_dist. The value at position IxJ indicates
               the index of the Jth smallest distance to the Ith query value in the
               train data array.
               the index of the Ith smallest distance of the Mth query.
   \param[out] dist is an array of MxN size, where M is equal to the number of query
               features and N is equal to n_dist. The value at position IxJ indicates
               the Hamming distance of the Jth smallest distance to the Ith query
               value in the train data array.
   \param[in]  query is the array containing the data to be queried
   \param[in]  train is the array containing the data used as training data
   \param[in]  dist_dim indicates the dimension to analyze for distance (the dimension
               indicated here must be of equal length for both query and train arrays)
   \param[in]  n_dist is the number of smallest distances to return (currently, only 1
               is supported)

   \ingroup cv_func_hamming_matcher
 */
AFAPI void hammingMatcher(array& idx, array& dist,
                          const array& query, const array& train,
                          const dim_t dist_dim=0, const unsigned n_dist=1);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
        C Interface for FAST feature detector

        \param[out] out struct containing arrays for x and y
                    coordinates and score, while array orientation is set to 0
                    as FAST does not compute orientation, and size is set to 1
                    as FAST does not compute multiple scales
        \param[in]  in array containing a grayscale image (color images are
                    not supported)
        \param[in]  thr FAST threshold for which a pixel of the circle around
                    the central pixel is considered to be greater or smaller
        \param[in]  arc_length length of arc (or sequential segment) to be
                    tested, must be within range [9-16]
        \param[in]  non_max performs non-maximal suppression if true
        \param[in]  feature_ratio maximum ratio of features to detect, the
                    maximum number of features is calculated by
                    feature_ratio * in.elements(). The maximum number of
                    features is not based on the score, instead, features
                    detected after the limit is reached are discarded
        \param[in]  edge is the length of the edges in the image to be
                    discarded by FAST (minimum is 3, as the radius of the
                    circle)

        \ingroup cv_func_fast
    */
    AFAPI af_err af_fast(af_features *out, const af_array in, const float thr, const unsigned arc_length, const bool non_max, const float feature_ratio, const unsigned edge);

    /**
        C Interface for ORB feature descriptor

        \param[out] feat af_features struct composed of arrays for x and y
                    coordinates, score, orientation and size of selected features
        \param[out] desc Nx8 array containing extracted descriptors, where N is the
                    number of selected features
        \param[in]  in array containing a grayscale image (color images are not
                    supported)
        \param[in]  fast_thr FAST threshold for which a pixel of the circle around
                    the central pixel is considered to be brighter or darker
        \param[in]  max_feat maximum number of features to hold (will only keep the
                    max_feat features with higher Harris responses)
        \param[in]  scl_fctr factor to downsample the input image, meaning that
                    each level will hold prior level dimensions divided by scl_fctr
        \param[in]  levels number of levels to be computed for the image pyramid
        \param[in]  blur_img blur image with a Gaussian filter with sigma=2 before
                    computing descriptors to increase robustness against noise if
                    true

        \ingroup cv_func_orb
    */
    AFAPI af_err af_orb(af_features *feat, af_array *desc, const af_array in, const float fast_thr, const unsigned max_feat, const float scl_fctr, const unsigned levels, const bool blur_img);

    /**
       C Interface wrapper for Hamming matcher

       \param[out] idx is an array of MxN size, where M is equal to the number of query
                   features and N is equal to n_dist. The value at position IxJ indicates
                   the index of the Jth smallest distance to the Ith query value in the
                   train data array.
                   the index of the Ith smallest distance of the Mth query.
       \param[out] dist is an array of MxN size, where M is equal to the number of query
                   features and N is equal to n_dist. The value at position IxJ indicates
                   the Hamming distance of the Jth smallest distance to the Ith query
                   value in the train data array.
       \param[in]  query is the array containing the data to be queried
       \param[in]  train is the array containing the data used as training data
       \param[in]  dist_dim indicates the dimension to analyze for distance (the dimension
                   indicated here must be of equal length for both query and train arrays)
       \param[in]  n_dist is the number of smallest distances to return (currently, only 1
                   is supported)

       \ingroup cv_func_hamming_matcher
    */
    AFAPI af_err af_hamming_matcher(af_array* idx, af_array* dist,
                                    const af_array query, const af_array train,
                                    const dim_t dist_dim, const unsigned n_dist);

#ifdef __cplusplus
}
#endif
