/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename Tk, typename Tv>
void sort0ByKeyIterative(Param<Tk> okey, Param<Tv> oval, bool isAscending);

template<typename Tk, typename Tv>
void sortByKeyBatched(Param<Tk> okey, Param<Tv> oval, const int dim,
                      bool isAscending);

template<typename Tk, typename Tv>
void sort0ByKey(Param<Tk> okey, Param<Tv> oval, bool isAscending);

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
