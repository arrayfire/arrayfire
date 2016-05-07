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
        template<typename Tk, typename Tv, bool isAscending>
        void sort0ByKeyIterative(Param<Tk> okey, Param<Tv> oval);

        template<typename Tk, typename Tv, bool isAscending>
        void sortByKeyBatched(Param<Tk> pKey, Param<Tv> pVal, const int dim);

        template<typename Tk, typename Tv, bool isAscending>
        void sort0ByKey(Param<Tk> okey, Param<Tv> oval);

    }
}
