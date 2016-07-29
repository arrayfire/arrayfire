/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>

namespace cuda
{
    namespace kernel
    {
        template<typename Tk, typename Tv>
        void sort0ByKeyIterative(Param<Tk> okey, Param<Tv> oval, bool isAscending);

        template<typename Tk, typename Tv>
        void sortByKeyBatched(Param<Tk> pKey, Param<Tv> pVal, const int dim, bool isAscending);

        template<typename Tk, typename Tv>
        void sort0ByKey(Param<Tk> okey, Param<Tv> oval, bool isAscending);

    }
}
