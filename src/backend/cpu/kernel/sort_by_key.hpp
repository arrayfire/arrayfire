/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
#include <err_cpu.hpp>

namespace cpu
{
namespace kernel
{

template<typename Tk, typename Tv>
void sort0ByKeyIterative(Array<Tk> okey, Array<Tv> oval, bool isAscending);

template<typename Tk, typename Tv>
void sortByKeyBatched(Array<Tk> okey, Array<Tv> oval, const int dim, bool isAscending);

template<typename Tk, typename Tv>
void sort0ByKey(Array<Tk> okey, Array<Tv> oval, bool isAscending);

}
}
