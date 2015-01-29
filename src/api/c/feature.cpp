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

af_err af_destroy_features(af_features feat)
{
    feat.n = 0;

    try {
        if (feat.x != 0) AF_CHECK(af_destroy_array(feat.x));
        if (feat.y != 0) AF_CHECK(af_destroy_array(feat.y));
        if (feat.score != 0) AF_CHECK(af_destroy_array(feat.score));
        if (feat.orientation != 0) AF_CHECK(af_destroy_array(feat.orientation));
        if (feat.size != 0) AF_CHECK(af_destroy_array(feat.size));
    }
    CATCHALL;
    return AF_SUCCESS;
}

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
        af_create_handle(&feat.x, 4, out_dims, f32);
        af_create_handle(&feat.y, 4, out_dims, f32);
        af_create_handle(&feat.score, 4, out_dims, f32);
        af_create_handle(&feat.orientation, 4, out_dims, f32);
        af_create_handle(&feat.size, 4, out_dims, f32);
    }

    features::features(af_features f)
    {
        feat.n = f.n;
        af_weak_copy(&feat.x, f.x);
        af_weak_copy(&feat.y, f.y);
        af_weak_copy(&feat.score, f.score);
        af_weak_copy(&feat.orientation, f.orientation);
        af_weak_copy(&feat.size, f.size);
    }

    features& features::operator= (const features& f)
    {
        if (this != &f) {
            setNumFeatures(f.getNumFeatures());
            setX(f.getX());
            setY(f.getY());
            setScore(f.getScore());
            setOrientation(f.getOrientation());
            setSize(f.getSize());
        }

        return *this;
    }

    features::~features()
    {
        af_destroy_features(feat);
    }

    size_t features::getNumFeatures() const
    {
        return feat.n;
    }

    array features::getX() const
    {
        return weakCopy(feat.x);
    }

    array features::getY() const
    {
        return weakCopy(feat.y);
    }

    array features::getScore() const
    {
        return weakCopy(feat.score);
    }

    array features::getOrientation() const
    {
        return weakCopy(feat.orientation);
    }

    array features::getSize() const
    {
        return weakCopy(feat.size);
    }

    void features::setNumFeatures(const size_t n)
    {
        feat.n = n;
    }

    void features::setX(const array x)
    {
        af_weak_copy(&feat.x, x.get());
    }

    void features::setX(const af_array x)
    {
        af_weak_copy(&feat.x, x);
    }

    void features::setY(const array y)
    {
        af_weak_copy(&feat.y, y.get());
    }

    void features::setY(const af_array y)
    {
        af_weak_copy(&feat.y, y);
    }

    void features::setScore(const array score)
    {
        af_weak_copy(&feat.score, score.get());
    }

    void features::setScore(const af_array score)
    {
        af_weak_copy(&feat.score, score);
    }

    void features::setOrientation(const array orientation)
    {
        af_weak_copy(&feat.orientation, orientation.get());
    }

    void features::setOrientation(const af_array orientation)
    {
        af_weak_copy(&feat.orientation, orientation);
    }

    void features::setSize(const array size)
    {
        af_weak_copy(&feat.size, size.get());
    }

    void features::setSize(const af_array size)
    {
        af_weak_copy(&feat.size, size);
    }

    af_features features::get()
    {
        af_features f;
        f.n = feat.n;
        af_weak_copy(&f.x, feat.x);
        af_weak_copy(&f.y, feat.y);
        af_weak_copy(&f.score, feat.score);
        af_weak_copy(&f.orientation, feat.orientation);
        af_weak_copy(&f.size, feat.size);

        return f;
    }

};
