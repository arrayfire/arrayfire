/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

//FIXME CPU backend doesn't required the following class implementation
//FFTPlanCache.hpp is not used while building CPU backend.
#ifndef AF_CPU
#include <boost/thread.hpp>
#include <common/FFTPlanCache.hpp>
#include <backend.hpp>
#include <platform.hpp>

typedef boost::shared_mutex smutex_t;
typedef boost::shared_lock<smutex_t> rlock_t;
typedef boost::unique_lock<smutex_t> wlock_t;

namespace common
{
static smutex_t gFFTMutexes[detail::DeviceManager::MAX_DEVICES];

template<class T, typename P>
void FFTPlanCache<T, P>::setMaxCacheSize(size_t size)
{
    wlock_t lock(gFFTMutexes[detail::getActiveDeviceId()]);
    mMaxCacheSize = size;
}

template<class T, typename P>
size_t FFTPlanCache<T, P>::getMaxCacheSize() const
{
    rlock_t lock(gFFTMutexes[detail::getActiveDeviceId()]);
    return mMaxCacheSize;
}

template<class T, typename P>
std::shared_ptr<P> FFTPlanCache<T, P>::find(const std::string& key) const
{
    std::shared_ptr<P> res;

    rlock_t lock(gFFTMutexes[detail::getActiveDeviceId()]);
    for(unsigned i=0; i<mCache.size(); ++i) {
        if (key == mCache[i].first) {
            res = mCache[i].second;
            break;
        }
    }

    return res;
}

template<class T, typename P>
void FFTPlanCache<T, P>::push(const std::string key, std::shared_ptr<P> plan)
{
    wlock_t lock(gFFTMutexes[detail::getActiveDeviceId()]);

    if (mCache.size()>=mMaxCacheSize)
        mCache.pop_back();

    mCache.push_front(plan_pair_t(key, plan));
}

template class FFTPlanCache<detail::PlanCache, detail::PlanType>;
}
#endif
