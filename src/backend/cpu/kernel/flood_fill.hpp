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
    using Point      = std::pair<uint, uint>;
    using Candidates = std::queue<Point>;

    const size_t numSeeds = x.dims().elements();
    const dim4 inDims     = in.dims();

    auto isInside = [&inDims](uint x, uint y) -> bool {
        return (x >= 0 && x < inDims[0] && y >= 0 && y < inDims[1]);
    };

    Candidates queue;
    {
        auto oit = begin(out);
        for (auto xit = begin(x), yit = begin(y);
             xit != end(x) && yit != end(y); ++xit, ++yit) {
            if (isInside(*xit, *yit)) {
                queue.emplace(*xit, *yit);
                oit.operator->()[(*xit) + (*yit) * inDims[0]] = T(2);
            }
        }
    }

    NeighborhoodIterator<T> inNeighborhood(in, dim4(1, 1, 0, 0));
    NeighborhoodIterator<T> endOfNeighborhood;
    NeighborhoodIterator<T> outNeighborhood(out, dim4(1, 1, 0, 0));

    while (!queue.empty()) {
        auto p = queue.front();

        inNeighborhood.setCenter(dim4(p.first, p.second, 0, 0));
        outNeighborhood.setCenter(dim4(p.first, p.second, 0, 0));

        while (inNeighborhood != endOfNeighborhood) {
            const dim4 offsetP = inNeighborhood.offset();
            const uint currx   = static_cast<uint>(p.first + offsetP[0]);
            const uint curry   = static_cast<uint>(p.second + offsetP[1]);

            if (isInside(currx, curry) && (*outNeighborhood == 0)) {
                // Current point is inside image boundaries and hasn't been
                // visited at all.
                if (*inNeighborhood >= lower && *inNeighborhood <= upper) {
                    // Current pixel is within threshold limits.
                    // Mark as valid and push on to the queue
                    *outNeighborhood = T(2);
                    queue.emplace(currx, curry);
                } else {
                    // Not valid pixel
                    *outNeighborhood = T(1);
                }
            }
            // Both input and output neighborhood iterators
            // should increment in lock step for this algorithm
            // to work correctly
            ++inNeighborhood;
            ++outNeighborhood;
        }
        queue.pop();
    }

    for (auto outIter = begin(out); outIter != end(out); ++outIter) {
        *outIter = (*outIter == T(2) ? newValue : T(0));
    }
}

}  // namespace kernel
}  // namespace cpu
