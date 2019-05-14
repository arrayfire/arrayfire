/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>

namespace common {

// will generate indexes to flip input array
// of size original dims according to axes specified in flip
void genFlipIndex(std::vector<af_seq> &index, const af::dim4 flip,
                  const af::dim4 original_dims) {
    if(index.size() < 4)
        index.resize(4);

    for(int i=0; i<4; ++i) {
        if(flip[i] != 0) {
            af_seq reversed = {(double)(original_dims[i] - 1), 0, -1};
            index[i] = reversed;
        } else {
            index[i] = af_span;
        }
    }
}


}  // namespace common
