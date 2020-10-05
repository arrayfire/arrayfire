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

typedef void * af_features;

#ifdef __cplusplus
namespace af
{
    class array;

    /// Represents a feature returned by a feature detector
    ///
    /// \ingroup arrayfire_class
    /// \ingroup features_group_features
    class AFAPI features {
    private:
        af_features feat;

    public:
        /// Default constructor. Creates a features object with new features
        features();

        /// Creates a features object with n features with undefined locations
        features(const size_t n);

        /// Creates a features object from a C af_features object
        features(af_features f);

        ~features();

        /// Copy assignment operator
        features& operator= (const features& other);

#if AF_API_VERSION >= 38
        /// Copy constructor
        features(const features &other);

#if AF_COMPILER_CXX_RVALUE_REFERENCES
        /// Move constructor
        features(features &&other);

        /// Move assignment operator
        features &operator=(features &&other);
#endif
#endif

        /// Returns  the number of features represented by this object
        size_t getNumFeatures() const;

        /// Returns an af::array which represents the x locations of a feature
        array getX() const;

        /// Returns an af::array which represents the y locations of a feature
        array getY() const;

        /// Returns an array with the score of the features
        array getScore() const;

        /// Returns an array with the orientations of the features
        array getOrientation() const;

        /// Returns an array that represents the size of the features
        array getSize() const;

        /// Returns the underlying C af_features object
        af_features get() const;
    };

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /// Creates a new af_feature object with \p num features
    ///
    /// \param[out] feat The new feature that will be created
    /// \param[in] num The number of features that will be in the new features
    ///                object
    /// \returns AF_SUCCESS if successful
    /// \ingroup features_group_features
    AFAPI af_err af_create_features(af_features *feat, dim_t num);

    /// Increases the reference count of the feature and all of its associated
    /// arrays
    ///
    /// \param[out] out The reference to the incremented array
    /// \param[in] feat The features object whose will be incremented
    ///                 object
    /// \returns AF_SUCCESS if successful
    /// \ingroup features_group_features
    AFAPI af_err af_retain_features(af_features *out, const af_features feat);

    /// Returns the number of features associated with this object
    ///
    /// \param[out] num The number of features in the object
    /// \param[in] feat The feature whose count will be returned
    /// \ingroup features_group_features
    AFAPI af_err af_get_features_num(dim_t *num, const af_features feat);

    /// Returns the x positions of the features
    ///
    /// \param[out] out An array with all x positions of the features
    /// \param[in] feat The features object
    /// \ingroup features_group_features
    AFAPI af_err af_get_features_xpos(af_array *out, const af_features feat);

    /// Returns the y positions of the features
    ///
    /// \param[out] out An array with all y positions of the features
    /// \param[in] feat The features object
    /// \ingroup features_group_features
    AFAPI af_err af_get_features_ypos(af_array *out, const af_features feat);

    /// Returns the scores of the features
    ///
    /// \param[out] score An array with scores of the features
    /// \param[in] feat The features object
    /// \ingroup features_group_features
    AFAPI af_err af_get_features_score(af_array *score, const af_features feat);

    /// Returns the orientations of the features
    ///
    /// \param[out] orientation An array with the orientations of the features
    /// \param[in] feat The features object
    /// \ingroup features_group_features
    AFAPI af_err af_get_features_orientation(af_array *orientation, const af_features feat);

    /// Returns the size of the features
    ///
    /// \param[out] size An array with the sizes of the features
    /// \param[in] feat The features object
    /// \ingroup features_group_features
    AFAPI af_err af_get_features_size(af_array *size, const af_features feat);

    /// Reduces the reference count of each of the features
    ///
    /// \param[in] feat The features object whose reference count will be
    ///                 reduced
    /// \ingroup features_group_features
    AFAPI af_err af_release_features(af_features feat);

#ifdef __cplusplus
}
#endif
