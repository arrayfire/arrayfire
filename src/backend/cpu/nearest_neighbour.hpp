/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace cpu
{

template<typename T, typename To>
void nearest_neighbour(Array<uint>& idx, Array<To>& dist,
                       const Array<T>& query, const Array<T>& train,
                       const uint dist_dim, const uint n_dist,
                       const af_match_type dist_type = AF_SSD);

}
