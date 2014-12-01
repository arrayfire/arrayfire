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

namespace af
{

    features::features()
    {
        feat = new af_features;
        feat->n = 0;
    }

    features::features(const size_t n)
    {
        feat = new af_features;
        feat->n = n;
        feat->x = array(n, f32).get();
        feat->y = array(n, f32).get();
        feat->score = array(n, f32).get();
        feat->orientation = array(n, f32).get();
        feat->size = array(n, f32).get();
    }

    features::features(af_features* f) : feat(f)
    {
        feat->n = f->n;
    }

    size_t features::getNumFeatures()
    {
        return feat->n;
    }

    array features::getX()
    {
        return array(feat->x);
    }

    array features::getY()
    {
        return array(feat->y);
    }

    array features::getScore()
    {
        return array(feat->score);
    }

    array features::getOrientation()
    {
        return array(feat->orientation);
    }

    array features::getSize()
    {
        return array(feat->size);
    }

    void features::setNumFeatures(const size_t n)
    {
        feat->n = n;
    }

    void features::setX(const array x)
    {
        feat->x = x.get();
    }

    void features::setX(const af_array x)
    {
        feat->x = x;
    }

    void features::setY(const array y)
    {
        feat->y = y.get();
    }

    void features::setY(const af_array y)
    {
        feat->y = y;
    }

    void features::setScore(const array score)
    {
        feat->score = score.get();
    }

    void features::setScore(const af_array score)
    {
        feat->score = score;
    }

    void features::setOrientation(const array orientation)
    {
        feat->orientation = orientation.get();
    }

    void features::setOrientation(const af_array orientation)
    {
        feat->orientation = orientation;
    }

    void features::setSize(const array size)
    {
        feat->size = size.get();
    }

    void features::setSize(const af_array size)
    {
        feat->size = size;
    }

    af_features * features::get()
    {
        return feat;
    }

};
