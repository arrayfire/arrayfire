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
#include <build_version.hpp>
#include <clfft.hpp>
#include <common/ArrayFireTypesIO.hpp>
#include <common/DefaultMemoryManager.hpp>
#include <common/Logger.hpp>
#include <common/Version.hpp>
#include <common/defines.hpp>
#include <common/host_memory.hpp>
#include <common/util.hpp>
#include <device_manager.hpp>
#include <err_opencl.hpp>
#include <errorcodes.hpp>
#include <af/opencl.h>
#include <af/version.h>
#include <memory>

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

using arrayfire::common::getEnvVar;
using cl::CommandQueue;
using cl::Context;
using cl::Device;
using cl::Platform;
using std::begin;
using std::end;
using std::find;
using std::make_unique;
using std::ostringstream;
using std::sort;
using std::string;
using std::stringstream;
using std::unique_ptr;
using std::vector;

namespace arrayfire {
namespace opencl {

#if defined(OS_MAC)
static const char* CL_GL_SHARING_EXT = "cl_APPLE_gl_sharing";
#else
static const char* CL_GL_SHARING_EXT = "cl_khr_gl_sharing";
#endif

bool checkExtnAvailability(const Device& pDevice, const string& pName) {
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

static afcl::deviceType getDeviceTypeEnum(const Device& dev) {
    return static_cast<afcl::deviceType>(dev.getInfo<CL_DEVICE_TYPE>());
}

static inline bool compare_default(const unique_ptr<Device>& ldev,
                                   const unique_ptr<Device>& rdev) {
    const cl_device_type device_types[] = {CL_DEVICE_TYPE_GPU,
                                           CL_DEVICE_TYPE_ACCELERATOR};

    auto l_dev_type = ldev->getInfo<CL_DEVICE_TYPE>();
    auto r_dev_type = rdev->getInfo<CL_DEVICE_TYPE>();

    // This ensures GPU > ACCELERATOR > CPU
    for (auto current_type : device_types) {
        auto is_l_curr_type = l_dev_type == current_type;
        auto is_r_curr_type = r_dev_type == current_type;

        if (is_l_curr_type && !is_r_curr_type) { return true; }
        if (!is_l_curr_type && is_r_curr_type) { return false; }
    }

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
                !verify_present(rPlatName, ref_name)) {
                return true;
            }

            if (!verify_present(lPlatName, ref_name) &&
                verify_present(rPlatName, ref_name)) {
                return false;
            }
        }

        // Intel falls back to compare based on memory
    } else {
        // If CPU, prefer Intel > AMD > POCL > APPLE
        const char* platforms[] = {"INTEL", "AMD", "POCL", "APPLE"};

        for (auto ref_name : platforms) {
            if (verify_present(lPlatName, ref_name) &&
                !verify_present(rPlatName, ref_name)) {
                return true;
            }

            if (!verify_present(lPlatName, ref_name) &&
                verify_present(rPlatName, ref_name)) {
                return false;
            }
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

        if (lres) { return true; }
        if (rres) { return false; }
    }

    // Default criteria, sort based on memory
    // Sort based on memory
    auto l_mem = ldev->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    auto r_mem = rdev->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    return l_mem > r_mem;
}

/// Class to compare two devices for sorting in a map
class deviceLess {
   public:
    bool operator()(const cl::Device& lhs, const cl::Device& rhs) const {
        return lhs() < rhs();
    }
};

DeviceManager::DeviceManager()
    : logger(common::loggerFactory("platform"))
    , mUserDeviceOffset(0)
    , fgMngr(nullptr)
    , mFFTSetup(new clfftSetupData) {
    vector<Platform> platforms;
    try {
        Platform::get(&platforms);
    } catch (const cl::Error& err) {
#if !defined(OS_MAC)
        // CL_PLATFORM_NOT_FOUND_KHR is not defined in Apple's OpenCL
        // implementation. Thus, it requires this ugly check.
        if (err.err() == CL_PLATFORM_NOT_FOUND_KHR) {
#endif
            AF_ERROR(
                "No OpenCL platforms found on this system. Ensure you have "
                "installed the device driver as well as the OpenCL runtime and "
                "ICD from your device vendor. You can use the clinfo utility "
                "to debug OpenCL installation issues.",
                AF_ERR_RUNTIME);
#if !defined(OS_MAC)
        }
#endif
    }
    fgMngr = std::make_unique<arrayfire::common::ForgeManager>();

    // This is all we need because the sort takes care of the order of devices
#ifdef OS_MAC
    cl_device_type DEVICE_TYPES = CL_DEVICE_TYPE_GPU;
#else
    cl_device_type DEVICE_TYPES = CL_DEVICE_TYPE_ALL;
#endif

    string deviceENV = getEnvVar("AF_OPENCL_DEVICE_TYPE");

    if (deviceENV == "GPU") {
        DEVICE_TYPES = CL_DEVICE_TYPE_GPU;
    } else if (deviceENV == "CPU") {
        DEVICE_TYPES = CL_DEVICE_TYPE_CPU;
    } else if (deviceENV.compare("ACC") >= 0) {
        DEVICE_TYPES = CL_DEVICE_TYPE_ACCELERATOR;
    }

    AF_TRACE("Found {} OpenCL platforms", platforms.size());

    std::map<cl::Device, cl::Context, deviceLess> mDeviceContextMap;
    // Iterate through platforms, get all available devices and store them
    for (auto& platform : platforms) {
        vector<Device> current_devices;

        try {
            platform.getDevices(DEVICE_TYPES, &current_devices);
        } catch (const cl::Error& err) {
            if (err.err() != CL_DEVICE_NOT_FOUND) { throw; }
        }
        AF_TRACE("Found {} devices on platform {}", current_devices.size(),
                 platform.getInfo<CL_PLATFORM_NAME>());
        if (!current_devices.empty()) {
            cl::Context ctx(current_devices);
            for (auto& dev : current_devices) {
                mDeviceContextMap[dev] = ctx;
                mDevices.emplace_back(make_unique<Device>(dev));
                AF_TRACE("Found device {} on platform {}",
                         dev.getInfo<CL_DEVICE_NAME>(),
                         platform.getInfo<CL_PLATFORM_NAME>());
            }
        }
    }

    int nDevices = mDevices.size();
    AF_TRACE("Found {} OpenCL devices", nDevices);

    if (nDevices == 0) { AF_ERROR("No OpenCL devices found", AF_ERR_RUNTIME); }

    // Sort OpenCL devices based on default criteria
    stable_sort(mDevices.begin(), mDevices.end(), compare_default);

    auto devices = move(mDevices);
    mDevices.clear();

    // Create contexts and queues once the sort is done
    for (int i = 0; i < nDevices; i++) {
        // For OpenCL-HPP >= v2023.12.14 type is cl::Platform instead of
        // cl_platform_id
        cl::Platform device_platform;
        device_platform = devices[i]->getInfo<CL_DEVICE_PLATFORM>();

        try {
            mContexts.emplace_back(
                make_unique<cl::Context>(mDeviceContextMap[*devices[i]]));
            mQueues.push_back(make_unique<CommandQueue>(
                *mContexts.back(), *devices[i], cl::QueueProperties::None));
            mIsGLSharingOn.push_back(false);
            mDeviceTypes.push_back(getDeviceTypeEnum(*devices[i]));
            mPlatforms.push_back(
                std::make_pair<std::unique_ptr<cl::Platform>, afcl_platform>(
                    make_unique<cl::Platform>(device_platform(), true),
                    getPlatformEnum(*devices[i])));
            mDevices.emplace_back(std::move(devices[i]));

            auto platform_version =
                mPlatforms.back().first->getInfo<CL_PLATFORM_VERSION>();
            string options;
            common::Version version =
                getOpenCLCDeviceVersion(*mDevices[i]).back();
#ifdef AF_WITH_FAST_MATH
            options = fmt::format(
                " -cl-std=CL{:Mm} -D dim_t={} -cl-fast-relaxed-math", version,
                dtype_traits<dim_t>::getName());
#else
            options = fmt::format(" -cl-std=CL{:Mm} -D dim_t={}", version,
                                  dtype_traits<dim_t>::getName());
#endif
            mBaseBuildFlags.push_back(options);
        } catch (const cl::Error& err) {
            AF_TRACE("Error creating context for device {} with error {}\n",
                     devices[i]->getInfo<CL_DEVICE_NAME>(), err.what());
        }
    }
    nDevices = mDevices.size();

    bool default_device_set = false;
    deviceENV               = getEnvVar("AF_OPENCL_DEFAULT_DEVICE");
    if (!deviceENV.empty()) {
        stringstream s(deviceENV);
        int def_device = -1;
        s >> def_device;
        if (def_device < 0 || def_device >= nDevices) {
            AF_TRACE(
                "AF_OPENCL_DEFAULT_DEVICE ({}) \
                   is out of range, Setting default device to 0",
                def_device);
        } else {
            setActiveContext(def_device);
            default_device_set = true;
        }
    }

    deviceENV = getEnvVar("AF_OPENCL_DEFAULT_DEVICE_TYPE");
    if (!default_device_set && !deviceENV.empty()) {
        cl_device_type default_device_type = CL_DEVICE_TYPE_GPU;
        if (deviceENV == "CPU") {
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
            AF_TRACE(
                "AF_OPENCL_DEFAULT_DEVICE_TYPE={} \
                   is not available, Using default device as 0",
                deviceENV);
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
            for (int i = 0; i < devCount; ++i) {
                markDeviceForInterop(i, wHandle);
            }
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
    for (auto& ctx : mContexts) {
        compute::context c(ctx->get());
        BoostProgCache currCache = compute::program_cache::get_global_cache(c);
        mBoostProgCacheVector.emplace_back(new BoostProgCache(currCache));
    }
    AF_TRACE("Default device: {}", getActiveDeviceId());
}

spdlog::logger* DeviceManager::getLogger() { return logger.get(); }

DeviceManager& DeviceManager::getInstance() {
    static auto* my_instance = new DeviceManager();
    return *my_instance;
}

void DeviceManager::setMemoryManager(
    std::unique_ptr<MemoryManagerBase> newMgr) {
    std::lock_guard<std::mutex> l(mutex);
    // It's possible we're setting a memory manager and the default memory
    // manager still hasn't been initialized, so initialize it anyways so we
    // don't inadvertently reset to it when we first call memoryManager()
    memoryManager();
    // Calls shutdown() on the existing memory manager.
    if (memManager) { memManager->shutdownAllocator(); }
    memManager = std::move(newMgr);
    // Set the backend memory manager for this new manager to register native
    // functions correctly.
    std::unique_ptr<opencl::Allocator> deviceMemoryManager(
        new opencl::Allocator());
    memManager->setAllocator(std::move(deviceMemoryManager));
    memManager->initialize();
}

void DeviceManager::resetMemoryManager() {
    // Replace with default memory manager
    std::unique_ptr<MemoryManagerBase> mgr(
        new common::DefaultMemoryManager(getDeviceCount(), common::MAX_BUFFERS,
                                         AF_MEM_DEBUG || AF_OPENCL_MEM_DEBUG));
    setMemoryManager(std::move(mgr));
}

void DeviceManager::setMemoryManagerPinned(
    std::unique_ptr<MemoryManagerBase> newMgr) {
    std::lock_guard<std::mutex> l(mutex);
    // It's possible we're setting a pinned memory manager and the default
    // memory manager still hasn't been initialized, so initialize it anyways so
    // we don't inadvertently reset to it when we first call
    // pinnedMemoryManager()
    pinnedMemoryManager();
    // Calls shutdown() on the existing memory manager.
    if (pinnedMemManager) { pinnedMemManager->shutdownAllocator(); }
    // Set the backend pinned memory manager for this new manager to register
    // native functions correctly.
    pinnedMemManager = std::move(newMgr);
    std::unique_ptr<opencl::AllocatorPinned> deviceMemoryManager(
        new opencl::AllocatorPinned());
    pinnedMemManager->setAllocator(std::move(deviceMemoryManager));
    pinnedMemManager->initialize();
}

void DeviceManager::resetMemoryManagerPinned() {
    // Replace with default memory manager
    std::unique_ptr<MemoryManagerBase> mgr(
        new common::DefaultMemoryManager(getDeviceCount(), common::MAX_BUFFERS,
                                         AF_MEM_DEBUG || AF_OPENCL_MEM_DEBUG));
    setMemoryManagerPinned(std::move(mgr));
}

DeviceManager::~DeviceManager() {
    for (int i = 0; i < getDeviceCount(); ++i) { gfxManagers[i] = nullptr; }
#ifndef OS_WIN
    // TODO: FIXME:
    // clfftTeardown() causes a "Pure Virtual Function Called" crash on
    // Windows for Intel devices. This causes tests to fail.
    clfftTeardown();
#endif

    deInitBlas();

    // deCache Boost program_cache
#ifndef OS_WIN
    for (auto bCache : mBoostProgCacheVector) { delete bCache; }
#endif

    memManager       = nullptr;
    pinnedMemManager = nullptr;

    // TODO: FIXME:
    // OpenCL libs on Windows platforms
    // are crashing the application at program exit
    // most probably a reference counting issue based
    // on the investigation done so far. This problem
    // doesn't seem to happen on Linux or MacOSX.
    // So, clean up OpenCL resources on non-Windows platforms
#ifdef OS_WIN
    for (auto& q : mQueues) { q.release(); }
    for (auto& c : mContexts) { c.release(); }
    for (auto& d : mDevices) { d.release(); }
#endif
}

void DeviceManager::markDeviceForInterop(const int device,
                                         const void* wHandle) {
    try {
        if (device >= static_cast<int>(mQueues.size()) ||
            device >= static_cast<int>(DeviceManager::MAX_DEVICES)) {
            AF_TRACE("Invalid device (}) passed for CL-GL Interop", device);
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
                static_cast<cl_context_properties>(wnd_ctx),
#if defined(_WIN32) || defined(_MSC_VER)
                CL_WGL_HDC_KHR,
                (cl_context_properties)wnd_dsp,
#else
                CL_GLX_DISPLAY_KHR,
                static_cast<cl_context_properties>(wnd_dsp),
#endif
                CL_CONTEXT_PLATFORM,
                (cl_context_properties)plat(),
                0
            };

            // Check if current OpenCL device is belongs to the OpenGL context
            {
                cl_context_properties test_cps[] = {
                    CL_GL_CONTEXT_KHR,
                    static_cast<cl_context_properties>(wnd_ctx),
                    CL_CONTEXT_PLATFORM, (cl_context_properties)plat(), 0};

                // Load the extension
                // If cl_khr_gl_sharing is available, this function should be
                // present This has been checked earlier, it comes to this point
                // only if it is found
                auto func = reinterpret_cast<clGetGLContextInfoKHR_fn>(
                    clGetExtensionFunctionAddressForPlatform(
                        plat(), "clGetGLContextInfoKHR"));

                // If the function doesn't load, bail early
                if (!func) { return; }

                // Get all devices associated with opengl context
                vector<cl_device_id> devices(16);
                size_t ret = 0;
                cl_int err = func(test_cps, CL_DEVICES_FOR_GL_CONTEXT_KHR,
                                  devices.size() * sizeof(cl_device_id),
                                  &devices[0], &ret);
                if (err != CL_SUCCESS) { return; }
                size_t num = ret / sizeof(cl_device_id);
                devices.resize(num);

                // Check if current device is present in the associated devices
                cl_device_id current_device = (*mDevices[device])();
                auto res = find(begin(devices), end(devices), current_device);

                if (res == end(devices)) { return; }
            }
#endif

            // Change current device to use GL sharing
            auto ctx = make_unique<Context>(*mDevices[device], cps);
            auto cq  = make_unique<CommandQueue>(*ctx, *mDevices[device]);

            mQueues[device]        = move(cq);
            mContexts[device]      = move(ctx);
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
}  // namespace arrayfire
