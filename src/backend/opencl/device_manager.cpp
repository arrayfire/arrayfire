/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Include this before af/opencl.h
// Causes conflict between system cl.hpp and opencl/cl.hpp
#include <common/graphics_common.hpp>

#include <GraphicsResourceManager.hpp>
#include <blas.hpp>
#include <clfft.hpp>
#include <common/defines.hpp>
#include <common/host_memory.hpp>
#include <common/util.hpp>
#include <device_manager.hpp>
#include <err_opencl.hpp>
#include <errorcodes.hpp>
#include <version.hpp>
#include <af/opencl.h>
#include <af/version.h>

#ifdef OS_MAC
#include <OpenGL/CGLCurrent.h>
#endif

#include <boost/compute/context.hpp>
#include <boost/compute/utility/program_cache.hpp>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using cl::CommandQueue;
using cl::Context;
using cl::Device;
using cl::Platform;
using std::begin;
using std::end;
using std::find;
using std::string;
using std::stringstream;
using std::vector;

namespace opencl {

#if defined(OS_MAC)
static const char* CL_GL_SHARING_EXT = "cl_APPLE_gl_sharing";
#else
static const char* CL_GL_SHARING_EXT = "cl_khr_gl_sharing";
#endif

bool checkExtnAvailability(const Device& pDevice, string pName) {
    bool ret_val = false;
    // find the extension required
    string exts = pDevice.getInfo<CL_DEVICE_EXTENSIONS>();
    stringstream ss(exts);
    string item;
    while (getline(ss, item, ' ')) {
        if (item == pName) {
            ret_val = true;
            break;
        }
    }
    return ret_val;
}

static afcl::deviceType getDeviceTypeEnum(Device dev) {
    return (afcl::deviceType)dev.getInfo<CL_DEVICE_TYPE>();
}

static inline bool compare_default(const Device* ldev, const Device* rdev) {
    const cl_device_type device_types[] = {CL_DEVICE_TYPE_GPU,
                                           CL_DEVICE_TYPE_ACCELERATOR};

    auto l_dev_type = ldev->getInfo<CL_DEVICE_TYPE>();
    auto r_dev_type = rdev->getInfo<CL_DEVICE_TYPE>();

    // This ensures GPU > ACCELERATOR > CPU
    for (auto current_type : device_types) {
        auto is_l_curr_type = l_dev_type == current_type;
        auto is_r_curr_type = r_dev_type == current_type;

        if (is_l_curr_type && !is_r_curr_type) return true;
        if (!is_l_curr_type && is_r_curr_type) return false;
    }

    // For GPUs, this ensures discrete > integrated
    auto is_l_integrated = ldev->getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>();
    auto is_r_integrated = rdev->getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>();

    if (!is_l_integrated && is_r_integrated) return true;
    if (is_l_integrated && !is_r_integrated) return false;

    // At this point, the devices are of same type.
    // Sort based on emperical evidence of preferred platforms

    // Prefer AMD first
    string lPlatName = getPlatformName(*ldev);
    string rPlatName = getPlatformName(*rdev);

    if (l_dev_type == CL_DEVICE_TYPE_GPU && r_dev_type == CL_DEVICE_TYPE_GPU) {
        // If GPU, prefer AMD > NVIDIA > Beignet / Intel > APPLE
        const char* platforms[] = {"AMD", "NVIDIA", "APPLE", "INTEL",
                                   "BEIGNET"};

        for (auto ref_name : platforms) {
            if (verify_present(lPlatName, ref_name) &&
                !verify_present(rPlatName, ref_name))
                return true;

            if (!verify_present(lPlatName, ref_name) &&
                verify_present(rPlatName, ref_name))
                return false;
        }

        // Intel falls back to compare based on memory
    } else {
        // If CPU, prefer Intel > AMD > POCL > APPLE
        const char* platforms[] = {"INTEL", "AMD", "POCL", "APPLE"};

        for (auto ref_name : platforms) {
            if (verify_present(lPlatName, ref_name) &&
                !verify_present(rPlatName, ref_name))
                return true;

            if (!verify_present(lPlatName, ref_name) &&
                verify_present(rPlatName, ref_name))
                return false;
        }
    }

    // Compare device compute versions

    {
        // Check Device OpenCL Version
        auto lversion = ldev->getInfo<CL_DEVICE_VERSION>();
        auto rversion = rdev->getInfo<CL_DEVICE_VERSION>();

        bool lres =
            (lversion[7] > rversion[7]) ||
            ((lversion[7] == rversion[7]) && (lversion[9] > rversion[9]));

        bool rres =
            (lversion[7] < rversion[7]) ||
            ((lversion[7] == rversion[7]) && (lversion[9] < rversion[9]));

        if (lres) return true;
        if (rres) return false;
    }

    // Default criteria, sort based on memory
    // Sort based on memory
    auto l_mem = ldev->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    auto r_mem = rdev->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    return l_mem > r_mem;
}

DeviceManager::DeviceManager()
    : mUserDeviceOffset(0)
    , fgMngr(new graphics::ForgeManager())
    , mFFTSetup(new clfftSetupData) {
    vector<Platform> platforms;
    Platform::get(&platforms);

    // This is all we need because the sort takes care of the order of devices
#ifdef OS_MAC
    cl_device_type DEVICE_TYPES = CL_DEVICE_TYPE_GPU;
#else
    cl_device_type DEVICE_TYPES = CL_DEVICE_TYPE_ALL;
#endif

    string deviceENV = getEnvVar("AF_OPENCL_DEVICE_TYPE");

    if (deviceENV.compare("GPU") == 0) {
        DEVICE_TYPES = CL_DEVICE_TYPE_GPU;
    } else if (deviceENV.compare("CPU") == 0) {
        DEVICE_TYPES = CL_DEVICE_TYPE_CPU;
    } else if (deviceENV.compare("ACC") >= 0) {
        DEVICE_TYPES = CL_DEVICE_TYPE_ACCELERATOR;
    }

    // Iterate through platforms, get all available devices and store them
    for (auto& platform : platforms) {
        vector<Device> current_devices;

        try {
            platform.getDevices(DEVICE_TYPES, &current_devices);
        } catch (const cl::Error& err) {
            if (err.err() != CL_DEVICE_NOT_FOUND) { throw; }
        }
        for (auto dev : current_devices) {
            mDevices.push_back(new Device(dev));
        }
    }

    int nDevices = mDevices.size();

    if (nDevices == 0) AF_ERROR("No OpenCL devices found", AF_ERR_RUNTIME);

    // Sort OpenCL devices based on default criteria
    stable_sort(mDevices.begin(), mDevices.end(), compare_default);

    // Create contexts and queues once the sort is done
    for (int i = 0; i < nDevices; i++) {
        cl_platform_id device_platform =
            mDevices[i]->getInfo<CL_DEVICE_PLATFORM>();
        cl_context_properties cps[3] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)(device_platform), 0};

        Context* ctx     = new Context(*mDevices[i], cps);
        CommandQueue* cq = new CommandQueue(*ctx, *mDevices[i]);
        mContexts.push_back(ctx);
        mQueues.push_back(cq);
        mIsGLSharingOn.push_back(false);
        mDeviceTypes.push_back(getDeviceTypeEnum(*mDevices[i]));
        mPlatforms.push_back(getPlatformEnum(*mDevices[i]));
    }

    bool default_device_set = false;
    deviceENV               = getEnvVar("AF_OPENCL_DEFAULT_DEVICE");
    if (!deviceENV.empty()) {
        stringstream s(deviceENV);
        int def_device = -1;
        s >> def_device;
        if (def_device < 0 || def_device >= (int)nDevices) {
            printf("WARNING: AF_OPENCL_DEFAULT_DEVICE is out of range\n");
            printf("Setting default device as 0\n");
        } else {
            setActiveContext(def_device);
            default_device_set = true;
        }
    }

    deviceENV = getEnvVar("AF_OPENCL_DEFAULT_DEVICE_TYPE");
    if (!default_device_set && !deviceENV.empty()) {
        cl_device_type default_device_type = CL_DEVICE_TYPE_GPU;
        if (deviceENV.compare("CPU") == 0) {
            default_device_type = CL_DEVICE_TYPE_CPU;
        } else if (deviceENV.compare("ACC") >= 0) {
            default_device_type = CL_DEVICE_TYPE_ACCELERATOR;
        }

        bool default_device_set = false;
        for (int i = 0; i < nDevices; i++) {
            if (mDevices[i]->getInfo<CL_DEVICE_TYPE>() == default_device_type) {
                default_device_set = true;
                setActiveContext(i);
                break;
            }
        }
        if (!default_device_set) {
            printf(
                "WARNING: AF_OPENCL_DEFAULT_DEVICE_TYPE=%s is not available\n",
                deviceENV.c_str());
            printf("Using default device as 0\n");
        }
    }

    // Define AF_DISABLE_GRAPHICS with any value to disable initialization
    string noGraphicsENV = getEnvVar("AF_DISABLE_GRAPHICS");
    if (fgMngr->plugin().isLoaded() && noGraphicsENV.empty()) {
        // If forge library was successfully loaded and
        // AF_DISABLE_GRAPHICS is not defined
        try {
            /* loop over devices and replace contexts with
             * OpenGL shared contexts whereever applicable */
            int devCount      = mDevices.size();
            fg_window wHandle = fgMngr->getMainWindow();
            for (int i = 0; i < devCount; ++i) markDeviceForInterop(i, wHandle);
        } catch (...) {}
    }

    mUserDeviceOffset = mDevices.size();
    // Initialize FFT setup data structure
    CLFFT_CHECK(clfftInitSetupData(mFFTSetup.get()));
    CLFFT_CHECK(clfftSetup(mFFTSetup.get()));

    // Initialize clBlas library
    initBlas();

    // Cache Boost program_cache
    namespace compute = boost::compute;
    for (auto ctx : mContexts) {
        compute::context c(ctx->get());
        BoostProgCache currCache = compute::program_cache::get_global_cache(c);
        mBoostProgCacheVector.emplace_back(new BoostProgCache(currCache));
    }
}

DeviceManager& DeviceManager::getInstance() {
    static DeviceManager* my_instance = new DeviceManager();
    return *my_instance;
}

DeviceManager::~DeviceManager() {
    for (int i = 0; i < getDeviceCount(); ++i) {
        delete gfxManagers[i].release();
    }
#ifndef OS_WIN
    // TODO: FIXME:
    // clfftTeardown() causes a "Pure Virtual Function Called" crash on
    // Windows for Intel devices. This causes tests to fail.
    clfftTeardown();
#endif

    deInitBlas();

    // deCache Boost program_cache
#ifndef OS_WIN
    namespace compute = boost::compute;
    for (auto bCache : mBoostProgCacheVector) delete bCache;
#endif

    delete memManager.release();
    delete pinnedMemManager.release();

    // TODO: FIXME:
    // OpenCL libs on Windows platforms
    // are crashing the application at program exit
    // most probably a reference counting issue based
    // on the investigation done so far. This problem
    // doesn't seem to happen on Linux or MacOSX.
    // So, clean up OpenCL resources on non-Windows platforms
#ifndef OS_WIN
    for (auto q : mQueues) delete q;
    for (auto c : mContexts) delete c;
    for (auto d : mDevices) delete d;
#endif
}

void DeviceManager::markDeviceForInterop(const int device,
                                         const void* wHandle) {
    try {
        if (device >= (int)mQueues.size() ||
            device >= (int)DeviceManager::MAX_DEVICES) {
            throw cl::Error(CL_INVALID_DEVICE,
                            "Invalid device passed for CL-GL Interop");
        } else {
            mQueues[device]->finish();

            // check if the device has CL_GL sharing extension enabled
            bool temp =
                checkExtnAvailability(*mDevices[device], CL_GL_SHARING_EXT);
            if (!temp) {
                /* return silently if given device has not OpenGL sharing
                 * extension enabled so that regular queue is used for it */
                return;
            }

            // call forge to get OpenGL sharing context and details
            Platform plat(mDevices[device]->getInfo<CL_DEVICE_PLATFORM>());

            long long wnd_ctx, wnd_dsp;
            fgMngr->plugin().fg_get_window_context_handle(
                &wnd_ctx, const_cast<fg_window>(wHandle));
            fgMngr->plugin().fg_get_window_display_handle(
                &wnd_dsp, const_cast<fg_window>(wHandle));
#ifdef OS_MAC
            CGLContextObj cgl_current_ctx = CGLGetCurrentContext();
            CGLShareGroupObj cgl_share_group =
                CGLGetShareGroup(cgl_current_ctx);

            cl_context_properties cps[] = {
                CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
                (cl_context_properties)cgl_share_group, 0};
#else
            cl_context_properties cps[] = {
                CL_GL_CONTEXT_KHR,
                (cl_context_properties)wnd_ctx,
#if defined(_WIN32) || defined(_MSC_VER)
                CL_WGL_HDC_KHR,
                (cl_context_properties)wnd_dsp,
#else
                CL_GLX_DISPLAY_KHR,
                (cl_context_properties)wnd_dsp,
#endif
                CL_CONTEXT_PLATFORM,
                (cl_context_properties)plat(),
                0
            };

            // Check if current OpenCL device is belongs to the OpenGL context
            {
                cl_context_properties test_cps[] = {
                    CL_GL_CONTEXT_KHR, (cl_context_properties)wnd_ctx,
                    CL_CONTEXT_PLATFORM, (cl_context_properties)plat(), 0};

                // Load the extension
                // If cl_khr_gl_sharing is available, this function should be
                // present This has been checked earlier, it comes to this point
                // only if it is found
                auto func = (clGetGLContextInfoKHR_fn)
                    clGetExtensionFunctionAddressForPlatform(
                        plat(), "clGetGLContextInfoKHR");

                // If the function doesn't load, bail early
                if (!func) return;

                // Get all devices associated with opengl context
                vector<cl_device_id> devices(16);
                size_t ret = 0;
                cl_int err = func(test_cps, CL_DEVICES_FOR_GL_CONTEXT_KHR,
                                  devices.size() * sizeof(cl_device_id),
                                  &devices[0], &ret);
                if (err != CL_SUCCESS) return;
                int num = ret / sizeof(cl_device_id);
                devices.resize(num);

                // Check if current device is present in the associated devices
                cl_device_id current_device = (*mDevices[device])();
                auto res = find(begin(devices), end(devices), current_device);

                if (res == end(devices)) return;
            }
#endif

            // Change current device to use GL sharing
            Context* ctx     = new Context(*mDevices[device], cps);
            CommandQueue* cq = new CommandQueue(*ctx, *mDevices[device]);

            // May be fixes the AMD GL issues we see on windows?
#if !defined(_WIN32) && !defined(_MSC_VER)
            delete mContexts[device];
            delete mQueues[device];
#endif

            mContexts[device]      = ctx;
            mQueues[device]        = cq;
            mIsGLSharingOn[device] = true;
        }
    } catch (const cl::Error& ex) {
        /* If replacing the original context with GL shared context
         * failes, don't throw an error and instead fall back to
         * original context and use copy via host to support graphics
         * on that particular OpenCL device. So mark it as no GL sharing */
    }
}

}  // namespace opencl
