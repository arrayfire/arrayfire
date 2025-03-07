/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <utility>

namespace arrayfire {
namespace oneapi {

template<typename Ti, typename To>
std::pair<Array<To>, Array<To>> sobelDerivatives(const Array<Ti> &img,
                                                 const unsigned &ker_size);

}  // namespace oneapi
}  // namespace arrayfire
