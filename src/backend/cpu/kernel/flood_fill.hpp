/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <ParamIterator.hpp>
#include <common/defines.hpp>

#include <queue>
#include <utility>

namespace arrayfire {
namespace cpu {
namespace kernel {

// Output array is set to the following values during the progression
// of the algorithm.
//
// 0 - not visited at all (default values in output because it was created
//                         using createValueArray helper at level of the
//                         functions caller)
// 1 - not valid
// 2 - valid (candidate for neighborhood walk, pushed onto the queue)
//
// Once, the algorithm is finished, output is reset
// to either zero or \p newValue for all valid pixels.
template<typename T>
void floodFill(Param<T> out, CParam<T> in, CParam<uint> x, CParam<uint> y,
               T newValue, T lower, T upper, af::connectivity connectivity) {
    UNUSED(connectivity);

    using af::dim4;
    using PtrDist    = typename ParamIterator<T>::difference_type;
    using Point      = std::pair<uint, uint>;
    using Candidates = std::queue<Point>;

    const dim4 dims    = in.dims();
    const dim4 strides = in.strides();

    ParamIterator<T> endOfNeighborhood;
    const dim4 nhoodRadii(1, 1, 0, 0);
    const dim4 nhood(2 * nhoodRadii[0] + 1, 2 * nhoodRadii[1] + 1,
                     2 * nhoodRadii[2] + 1, 2 * nhoodRadii[3] + 1);

    auto isInside = [&dims](uint x, uint y) {
        return (x >= 0 && x < dims[0] && y >= 0 && y < dims[1]);
    };
    auto leftTopPtr = [&strides, &nhoodRadii](T* ptr, const af::dim4& center) {
        T* ltPtr = ptr;
        for (dim_t d = 0; d < AF_MAX_DIMS; ++d) {
            ltPtr += ((center[d] - nhoodRadii[d]) * strides[d]);
        }
        return ltPtr;
    };
    Candidates queue;
    {
        auto oit = begin(out);
        for (auto xit = begin(x), yit = begin(y);
             xit != end(x) && yit != end(y); ++xit, ++yit) {
            if (isInside(*xit, *yit)) {
                queue.emplace(*xit, *yit);
                oit.operator->()[(*xit) + (*yit) * dims[0]] = T(2);
            }
        }
    }

    T* inPtr  = const_cast<T*>(in.get());
    T* outPtr = out.get();

    while (!queue.empty()) {
        Point& p = queue.front();

        const dim4 center(p.first, p.second, 0, 0);

        CParam<T> inNHood(const_cast<const T*>(leftTopPtr(inPtr, center)),
                          nhood, strides);
        Param<T> outNHood(leftTopPtr(outPtr, center), nhood, strides);

        ParamIterator<T> inIter(inNHood);
        ParamIterator<T> outIter(outNHood);

        while (inIter != endOfNeighborhood) {
            const T* ptr     = inIter.operator->();
            PtrDist dist     = ptr - inPtr;
            const uint currx = static_cast<uint>(dist % dims[0]);
            const uint curry = static_cast<uint>(dist / dims[0]);

            if (isInside(currx, curry) && (*outIter == 0)) {
                // Current point is inside image boundaries and hasn't been
                // visited at all.
                if (*inIter >= lower && *inIter <= upper) {
                    // Current pixel is within threshold limits.
                    // Mark as valid and push on to the queue
                    *outIter = T(2);
                    queue.emplace(currx, curry);
                } else {
                    // Not valid pixel
                    *outIter = T(1);
                }
            }
            // Both input and output neighborhood iterators
            // should increment in lock step for this algorithm
            // to work correctly
            ++inIter;
            ++outIter;
        }
        queue.pop();
    }

    for (auto outIter = begin(out); outIter != end(out); ++outIter) {
        *outIter = (*outIter == T(2) ? newValue : T(0));
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
