/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/features.h>
#include <af/array.h>
#include <handle.hpp>
#include "error.hpp"

namespace af
{

    features::features()
    {
        AF_THROW(af_create_features(&feat, 0));
    }

    features::features(const size_t n)
    {
        AF_THROW(af_create_features(&feat, (int)n));
    }

    features::features(af_features f) : feat(f)
    {
    }

    features& features::operator= (const features& other)
    {
        if (this != &other) {
            AF_THROW(af_release_features(feat));
            AF_THROW(af_retain_features(&feat, other.get()));
        }
        return *this;
    }

    features::~features()
    {
        if(AF_SUCCESS != af_release_features(feat)) {
            fprintf(stderr, "Error: Couldn't release af::features: %p\n", this);
        }
    }

    size_t features::getNumFeatures() const
    {
        dim_t n = 0;
        AF_THROW(af_get_features_num(&n, feat));
        return n;
    }

    array features::getX() const
    {
        af_array x = 0;
        AF_THROW(af_get_features_xpos(&x, feat));
        af_array tmp = 0;
        AF_THROW(af_retain_array(&tmp, x));
        return array(tmp);
    }

    array features::getY() const
    {
        af_array y = 0;
        AF_THROW(af_get_features_ypos(&y, feat));
        af_array tmp = 0;
        AF_THROW(af_retain_array(&tmp, y));
        return array(tmp);
    }

    array features::getScore() const
    {
        af_array s = 0;
        AF_THROW(af_get_features_score(&s, feat));
        af_array tmp = 0;
        AF_THROW(af_retain_array(&tmp, s));
        return array(tmp);
    }

    array features::getOrientation() const
    {
        af_array ori = 0;
        AF_THROW(af_get_features_orientation(&ori, feat));
        af_array tmp = 0;
        AF_THROW(af_retain_array(&tmp, ori));
        return array(tmp);
    }

    array features::getSize() const
    {
        af_array s = 0;
        AF_THROW(af_get_features_size(&s, feat));
        af_array tmp = 0;
        AF_THROW(af_retain_array(&tmp, s));
        return array(tmp);
    }

    af_features features::get() const
    {
        return feat;
    }

};
