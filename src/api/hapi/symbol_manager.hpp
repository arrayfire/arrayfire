/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <functional>
#include <string>
#include <stdlib.h>
#include <dlfcn.h>

class AFSymbolManager {
    public:
        static AFSymbolManager& getInstance();

        ~AFSymbolManager();

        void setBackend(af::Backend bnkd);

        template<typename... CalleeArgs>
        af_err call(const char* symbolName, CalleeArgs... args) {
            using std::string;
            using std::logic_error;

            void* const handle = dlsym(activeHandle, symbolName);

            if (!handle) {
                char* const error = dlerror();
                if (error) {
                    throw logic_error("can't find symbol: "+string(symbolName)+" - "+error);
                }
            }

            std::function<af_err (CalleeArgs...)> callee = reinterpret_cast<af_err (*)(CalleeArgs...)>(handle);

            return callee(args...);
        }

    protected:
        AFSymbolManager();

        // Following two declarations are required to
        // avoid copying accidental copy/assignment
        // of instance returned by getInstance to other
        // variables
        AFSymbolManager(AFSymbolManager const&);
        void operator=(AFSymbolManager const&);

    private:
        bool isCPULoaded;
        bool isCUDALoaded;
        bool isOCLLoaded;

        void* cpuHandle;
        void* cudaHandle;
        void* oclHandle;

        af::Backend activeBknd;
        void* activeHandle;
};
