/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <deque>
#include <string>
#include <utility>

namespace common
{
// FFTPlanCache caches backend specific fft plans
//
// new plan |--> IF number of plans cached is at limit, pop the least used entry and push new plan.
//          |
//          |--> ELSE just push the plan
// existing plan -> reuse a plan
template<typename T, typename P>
class FFTPlanCache
{
    public:
        FFTPlanCache() : mMaxCacheSize(5) {
            static_cast<T*>(this)->initLibrary();
        }

        ~FFTPlanCache() {
            static_cast<T*>(this)->deInitLibrary();
        }

        inline void maxCacheSize(size_t size) {
            mCache.resize(size, std::make_pair<std::string, P>(std::string(""), 0));
        }

        inline size_t maxCacheSize() const {
            return mMaxCacheSize;
        }

        inline P get(int index) const {
            return mCache[index].second;
        }

        // iterates through plan cache from front to back
        // of the cache(queue)
        //
        // A valid index of the plan in the cache is returned
        // otherwise -1 is returned
        int find(std::string key) const {
            int retVal = -1;
            for(uint i=0; i<mCache.size(); ++i) {
                if (key == mCache[i].first) {
                    retVal = i;
                }
            }
            return retVal;
        }

        // pops plan from the back of cache(queue)
        void pop() {
            if (!mCache.empty()) {
                // destroy the cufft plan associated with the
                // least recently used plan
                static_cast<T*>(this)->removePlan(mCache.back().second);
                // now pop the entry from cache
                mCache.pop_back();
            }
        }

        // pushes plan to the front of cache(queue)
        void push(std::string key, P plan) {
            if (mCache.size()>mMaxCacheSize) {
                pop();
            }
            mCache.push_front(std::pair<std::string, P>(key, plan));
        }

    private:
        FFTPlanCache(FFTPlanCache const&);
        void operator=(FFTPlanCache const&);

        size_t       mMaxCacheSize;

        std::deque< std::pair<std::string, P> > mCache;

};
}
