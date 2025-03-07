/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename Tk, typename Tv>
void sort0ByKeyIterative(Param<Tk> pKey, Param<Tv> pVal, bool isAscending);

template<typename Tk, typename Tv>
void sortByKeyBatched(Param<Tk> pKey, Param<Tv> pVal, const int dim,
                      bool isAscending);

template<typename Tk, typename Tv>
void sort0ByKey(Param<Tk> pKey, Param<Tv> pVal, bool isAscending);

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
