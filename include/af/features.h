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

    class AFAPI features {
    private:
        af_features feat;

    public:
        features();
        features(const size_t n);
        features(af_features f);

        ~features();

        features& operator= (const features& f);

        size_t getNumFeatures() const;
        array getX() const;
        array getY() const;
        array getScore() const;
        array getOrientation() const;
        array getSize() const;
        af_features get() const;
    };

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err af_create_features(af_features *feat, dim_t num);

    AFAPI af_err af_retain_features(af_features *out, const af_features feat);

    AFAPI af_err af_get_features_num(dim_t *num, const af_features feat);

    AFAPI af_err af_get_features_xpos(af_array *out, const af_features feat);

    AFAPI af_err af_get_features_ypos(af_array *out, const af_features feat);

    AFAPI af_err af_get_features_score(af_array *score, const af_features feat);

    AFAPI af_err af_get_features_orientation(af_array *orientation, const af_features feat);

    AFAPI af_err af_get_features_size(af_array *size, const af_features feat);

    // Destroy af_features
    AFAPI af_err af_release_features(af_features feat);

#ifdef __cplusplus
}
#endif
