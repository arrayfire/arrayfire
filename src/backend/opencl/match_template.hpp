/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace opencl {

template<typename inType, typename outType, af_match_type mType>
Array<outType> match_template(const Array<inType> &sImg,
                              const Array<inType> &tImg);

}
