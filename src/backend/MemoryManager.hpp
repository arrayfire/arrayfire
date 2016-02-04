/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <vector>
#include <map>
#include <mutex>

namespace common
{

typedef std::recursive_mutex mutex_t;
typedef std::lock_guard<mutex_t> lock_guard_t;

const unsigned MAX_BUFFERS   = 1000;
const size_t ONE_GB = 1 << 30;

class MemoryManager
{
    typedef struct
    {
        bool manager_lock;
        bool user_lock;
        size_t bytes;
    } locked_info;

    typedef std::map<void *, locked_info> locked_t;
    typedef locked_t::iterator locked_iter;

    typedef std::map<size_t, std::vector<void *> >free_t;
    typedef free_t::iterator free_iter;

    typedef struct
    {
        locked_t locked_map;
        free_t   free_map;

        size_t lock_bytes;
        size_t lock_buffers;
        size_t total_bytes;
        size_t total_buffers;
        size_t max_bytes;
    } memory_info;

    size_t mem_step_size;
    unsigned max_buffers;
    std::vector<memory_info> memory;
    bool debug_mode;

    memory_info& getCurrentMemoryInfo()
    {
        return memory[this->getActiveDeviceId()];
    }

    virtual int getActiveDeviceId()
    {
        return 0;
    }

    virtual size_t getMaxMemorySize(int id)
    {
        return 0;
    }

public:
    MemoryManager(int num_devices, unsigned MAX_BUFFERS, bool debug);

    void setMaxMemorySize();

    void *alloc(const size_t bytes, bool user_lock);

    void unlock(void *ptr, bool user_unlock);

    void garbageCollect();

    void printInfo(const char *msg, const int device);

    void bufferInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                    size_t *lock_bytes,  size_t *lock_buffers);

    void userLock(const void *ptr);

    void userUnlock(const void *ptr);

    size_t getMemStepSize();

    size_t getMaxBytes();

    unsigned getMaxBuffers();

    void setMemStepSize(size_t new_step_size);

    virtual void *nativeAlloc(const size_t bytes)
    {
        return malloc(bytes);
    }

    virtual void nativeFree(void *ptr)
    {
        free((void *)ptr);
    }

    virtual ~MemoryManager()
    {
    }

protected:
    mutex_t memory_mutex;

};

}
