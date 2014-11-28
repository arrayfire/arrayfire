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
#include <af/array.h>
#include <af/dim4.hpp>
#include <af/seq.h>
#include <af/traits.hpp>


#ifdef __cplusplus
namespace af
{

    class AFAPI features {
    private:
        af_features * feat;

    public:
        features();
        features(const size_t n);
        features(af_features* f);

        size_t getNumFeatures();
        af_array getX();
        af_array getY();
        af_array getScore();
        af_array getOrientation();
        af_array getSize();

        void setNumFeatures(const size_t n);
        void setX(const af_array x);
        void setY(const af_array y);
        void setScore(const af_array score);
        void setOrientation(const af_array orientation);
        void setSize(const af_array size);

        af_features * get();
    };

}
#endif
