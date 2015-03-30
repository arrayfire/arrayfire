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
        feat.n = 0;
        feat.x = 0;
        feat.y = 0;
        feat.score = 0;
        feat.orientation = 0;
        feat.size = 0;
    }

    features::features(const size_t n)
    {
        feat.n = n;
        dim_type out_dims[4] = {dim_type(n), 1, 1, 1};
        AF_THROW(af_create_handle(&feat.x, 4, out_dims, f32));
        AF_THROW(af_create_handle(&feat.y, 4, out_dims, f32));
        AF_THROW(af_create_handle(&feat.score, 4, out_dims, f32));
        AF_THROW(af_create_handle(&feat.orientation, 4, out_dims, f32));
        AF_THROW(af_create_handle(&feat.size, 4, out_dims, f32));
    }

    features::features(af_features f) : feat(f)
    {
    }

    features& features::operator= (const features& other)
    {
        if (this != &other) {

            if (feat.n > 0) {
                AF_THROW(af_destroy_features(feat));
            }

            af_features f = other.get();

            feat.n = f.n;
            if (f.n > 0) {
                AF_THROW(af_weak_copy(&feat.x, f.x));
                AF_THROW(af_weak_copy(&feat.y, f.y));
                AF_THROW(af_weak_copy(&feat.score, f.score));
                AF_THROW(af_weak_copy(&feat.orientation, f.orientation));
                AF_THROW(af_weak_copy(&feat.size, f.size));
            } else {
                feat.x = 0;
                feat.y = 0;
                feat.score = 0;
                feat.orientation = 0;
                feat.size = 0;
            }
        }

        return *this;
    }

    features::~features()
    {
        if (feat.n >= 0) {
            if(AF_SUCCESS != af_destroy_features(feat)) {
                fprintf(stderr, "Error: Couldn't destroy af::features: %p\n", this);
            }
        }
    }

    size_t features::getNumFeatures() const
    {
        return feat.n;
    }

    array features::getX() const
    {
        if (feat.n == 0) return array();
        af_array tmp = 0;
        AF_THROW(af_weak_copy(&tmp, feat.x));
        return array(tmp);
    }

    array features::getY() const
    {
        if (feat.n == 0) return array();
        af_array tmp = 0;
        AF_THROW(af_weak_copy(&tmp, feat.y));
        return array(tmp);
    }

    array features::getScore() const
    {
        if (feat.n == 0) return array();
        af_array tmp = 0;
        AF_THROW(af_weak_copy(&tmp, feat.score));
        return array(tmp);
    }

    array features::getOrientation() const
    {
        if (feat.n == 0) return array();
        af_array tmp = 0;
        AF_THROW(af_weak_copy(&tmp, feat.orientation));
        return array(tmp);
    }

    array features::getSize() const
    {
        if (feat.n == 0) return array();
        af_array tmp = 0;
        AF_THROW(af_weak_copy(&tmp, feat.size));
        return array(tmp);
    }

    af_features features::get() const
    {
        return feat;
    }

};
