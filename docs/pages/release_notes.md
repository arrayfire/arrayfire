Release Notes {#releasenotes}
==============

v3.8.3
======

## Improvements

- Add support for CUDA 12 \PR{3352}
- Modernize documentation style and content \PR{3351}
- memcpy performance improvements \PR{3144}
- JIT performance improvements \PR{3144}
- join performance improvements \PR{3144}
- Improve support for Intel and newer Clang compilers \PR{3334}
- CCache support on Windows \PR{3257}

## Fixes

- Fix issue with some locales with OpenCL kernel generation \PR{3294}
- Internal improvements
- Fix leak in clfft on exit.
- Fix some cases where ndims was incorrectly used ot calculate shape \PR{3277}
- Fix issue when setDevice was not called in new threads \PR{3269}
- Restrict initializer list to just fundamental types \PR{3264}

## Contributions

Special thanks to our contributors:
[Carlo Cabrera](https://github.com/carlocab)
[Guillaume Schmid](https://github.com/GuillaumeSchmid)
[Willy Born](https://github.com/willyborn)
[ktdq](https://github.com/ktdq)

v3.8.2
======

## Improvements

- Optimize JIT by removing some consecutive cast operations \PR{3031}
- Add driver checks checks for CUDA 11.5 and 11.6 \PR{3203}
- Improve the timing algorithm used for timeit \PR{3185}
- Dynamically link against CUDA numeric libraries by default \PR{3205}
- Add support for pruning CUDA binaries to reduce static binary sizes \PR{3234} \PR{3237}
- Remove unused cuDNN libraries from installations \PR{3235}
- Add support to staticly link NVRTC libraries after CUDA 11.5 \PR{3236}
- Add support for compiling with ccache when building the CUDA backend \PR{3241}

## Fixes

- Fix issue with consecutive moddims operations in the CPU backend \PR{3232}
- Better floating point comparisons for tests \PR{3212}
- Fix several warnings and inconsistencies with doxygen and documentation \PR{3226}
- Fix issue when passing empty arrays into join \PR{3211}
- Fix default value for the `AF_COMPUTE_LIBRARY` when not set \PR{3228}
- Fix missing symbol issue when MKL is staticly linked \PR{3244}
- Remove linking of OpenCL's library to the unified backend \PR{3244}

## Contributions

Special thanks to our contributors:
[Jacob Kahn](https://github.com/jacobkahn)
[Willy Born](https://github.com/willyborn)

v3.8.1
======

## Improvements

- moddims now uses JIT approach for certain special cases - \PR{3177}
- Embed Version Info in Windows DLLs - \PR{3025} 
- OpenCL device max parameter is now queries from device properties - \PR{3032} 
- JIT Performance Optimization: Unique funcName generation sped up - \PR{3040} 
- Improved readability of log traces  - \PR{3050} 
- Use short function name in non-debug build error messages - \PR{3060} 
- SIFT/GLOH are now available as part of website binaries - \PR{3071} 
- Short-circuit zero elements case in detail::copyArray backend function - \PR{3059} 
- Speedup of kernel caching mechanism - \PR{3043} 
- Add short-circuit check for empty Arrays in JIT evalNodes - \PR{3072} 
- Performance optimization of indexing using dynamic thread block sizes - \PR{3111} 
- ArrayFire starting with this release will use Intel MKL single dynamic library which resolves lot of linking issues unified library had when user applications used MKL themselves - \PR{3120} 
- Add shortcut check for zero elements in af_write_array - \PR{3130} 
- Speedup join by eliminating temp buffers for cascading joins - \PR{3145} 
- Added batch support for solve - \PR{1705} 
- Use pinned memory to copy device pointers in CUDA solve - \PR{1705} 
- Added package manager instructions to docs - \PR{3076} 
- CMake Build Improvements - \PR{3027} , \PR{3089} , \PR{3037} , \PR{3072} , \PR{3095} , \PR{3096} , \PR{3097} , \PR{3102} , \PR{3106} , \PR{3105} , \PR{3120} , \PR{3136} , \PR{3135} , \PR{3137} , \PR{3119} , \PR{3150} , \PR{3138} , \PR{3156} , \PR{3139} , \PR{1705} , \PR{3162} 
- CPU backend improvements - \PR{3010} , \PR{3138} , \PR{3161} 
- CUDA backend improvements - \PR{3066} , \PR{3091} , \PR{3093} , \PR{3125} , \PR{3143} , \PR{3161} 
- OpenCL backend improvements - \PR{3091} , \PR{3068} , \PR{3127} , \PR{3010} , \PR{3039} , \PR{3138} , \PR{3161} 
- General(including JIT) performance improvements across backends - \PR{3167} 
- Testing improvements - \PR{3072} , \PR{3131} , \PR{3151} , \PR{3141} , \PR{3153} , \PR{3152} , \PR{3157} , \PR{1705} , \PR{3170} , \PR{3167} 
- Update CLBlast to latest version - \PR{3135} , \PR{3179} 
- Improved Otsu threshold computation helper in canny algorithm - \PR{3169} 
- Modified default parameters for fftR2C and fftC2R C++ API from 0 to 1.0 - \PR{3178} 
- Use appropriate MKL getrs_batch_strided API based on MKL Versions - \PR{3181} 

## Fixes

- Fixed a bug JIT kernel disk caching - \PR{3182} 
- Fixed stream used by thrust(CUDA backend) functions - \PR{3029}  
- Added workaround for new cuSparse API that was added by CUDA amid fix releases - \PR{3057} 
- Fixed `const` array indexing inside `gfor` - \PR{3078} 
- Handle zero elements in copyData to host - \PR{3059} 
- Fixed double free regression in OpenCL backend - \PR{3091} 
- Fixed an infinite recursion bug in NaryNode JIT Node - \PR{3072} 
- Added missing input validation check in sparse-dense arithmetic operations - \PR{3129} 
- Fixed bug in `getMappedPtr` in OpenCL due to invalid lambda capture - \PR{3163} 
- Fixed bug in `getMappedPtr` on Arrays that are not ready - \PR{3163} 
- Fixed edgeTraceKernel for CPU devices on OpenCL backend - \PR{3164} 
- Fixed windows build issue(s) with VS2019 - \PR{3048}
- API documentation fixes - \PR{3075} , \PR{3076} , \PR{3143} , \PR{3161} 
- CMake Build Fixes - \PR{3088} 
- Fixed the tutorial link in README - \PR{3033} 
- Fixed function name typo in timing tutorial - \PR{3028} 
- Fixed couple of bugs in CPU backend canny implementation - \PR{3169} 
- Fixed reference count of array(s) used in JIT operations. It is related to arrayfire's internal memory book keeping. The behavior/accuracy of arrayfire code wasn't broken earlier. It corrected the reference count to be of optimal value in the said scenarios. This may potentially reduce memory usage in some narrow cases - \PR{3167} 
- Added assert that checks if topk is called with a negative value for k - \PR{3176} 
- Fixed an Issue where countByKey would give incorrect results for any n > 128 - \PR{3175} 

## Contributions

Special thanks to our contributors: [HO-COOH][1], [Willy Born][2], [Gilad Avidov][3], [Pavan Yalamanchili][4]

[1]: https://github.com/HO-COOH  
[2]: https://github.com/willyborn  
[3]: https://github.com/avidov  
[4]: https://github.com/pavanky  


v3.8.0
======

Major Updates
--------
- Non-uniform(ragged) reductions \PR{2786}
- Bit-wise not operator support for array and C API (af\_bitnot) \PR{2865}
- Initialization list constructor for array class \PR{2829} \PR{2987}

Improvements
------------
- New API for following statistics function: cov, var and stdev - \PR{2986}
- allocV2 and freeV2 which return cl\_mem on OpenCL backend \PR{2911}
- Move constructor and move assignment operator for Dim4 class \PR{2946}
- Support for CUDA 11.1 and Compute 8.6 \PR{3023}
- Fix af::feature copy constructor for multi-threaded sceanarios \PR{3022}

v3.7.3
======

Improvements
------------
- Add f16 support for histogram - \PR{2984}
- Update confidence connected components example with better illustration - \PR{2968}
- Enable disk caching of OpenCL kernel binaries - \PR{2970}
- Refactor extension of kernel binaries stored to disk `.bin` - \PR{2970}
- Add minimum driver versions for CUDA toolkit 11 in internal map - \PR{2982}
- Improve warnings messages from run-time kernel compilation functions - \PR{2996}

Fixes
-----
- Fix bias factor of variance in var_all and cov functions - \PR{2986}
- Fix a race condition in confidence connected components function for OpenCL backend - \PR{2969}
- Safely ignore disk cache failures in CUDA backend for compiled kernel binaries - \PR{2970}
- Fix randn by passing in correct values to Box-Muller - \PR{2980}
- Fix rounding issues in Box-Muller function used for RNG - \PR{2980}
- Fix problems in RNG for older compute architectures with fp16 - \PR{2980}  \PR{2996}
- Fix performance regression of approx functions - \PR{2977}
- Remove assert that check that signal/filter types have to be the same - \PR{2993}
- Fix `checkAndSetDevMaxCompute` when the device cc is greater than max - \PR{2996}
- Fix documentation errors and warnings - \PR{2973} , \PR{2987}
- Add missing opencl-arrayfire interoperability functions in unified backend  - \PR{2981}

Contributions
-------------
Special thanks to our contributors:
[P. J. Reed](https://github.com/pjreed)

v3.7.2
======

Improvements
------------
- Cache CUDA kernels to disk to improve load times(Thanks to \@cschreib-ibex) \PR{2848}
- Staticly link against cuda libraries \PR{2785}
- Make cuDNN an optional build dependency \PR{2836}
- Improve support for different compilers and OS \PR{2876} \PR{2945} \PR{2925} \PR{2942} \PR{2943} \PR{2945} \PR{2958}
- Improve performance of join and transpose on CPU \PR{2849}
- Improve documentation \PR{2816} \PR{2821} \PR{2846} \PR{2918} \PR{2928} \PR{2947}
- Reduce binary size using NVRTC and template reducing instantiations \PR{2849} \PR{2861} \PR{2890} \PR{2957}
- reduceByKey performance improvements \PR{2851} \PR{2957}
- Improve support for Intel OpenCL GPUs \PR{2855}
- Allow staticly linking against MKL \PR{2877} (Sponsered by SDL)
- Better support for older CUDA toolkits \PR{2923}
- Add support for CUDA 11 \PR{2939}
- Add support for ccache for faster builds \PR{2931}
- Add support for the conan package manager on linux \PR{2875}
- Propagate build errors up the stack in AFError exceptions \PR{2948} \PR{2957}
- Improve runtime dependency library loading \PR{2954}
- Improved cuDNN runtime checks and warnings \PR{2960}
- Document af\_memory\_manager\_* native memory return values \PR{2911}

Fixes
-----
- Bug crash when allocating large arrays \PR{2827}
- Fix various compiler warnings \PR{2827} \PR{2849} \PR{2872} \PR{2876}
- Fix minor leaks in OpenCL functions \PR{2913}
- Various continuous integration related fixes \PR{2819}
- Fix zero padding with convolv2NN \PR{2820}
- Fix af_get_memory_pressure_threshold return value \PR{2831}
- Increased the max filter length for morph
- Handle empty array inputs for LU, QR, and Rank functions \PR{2838}
- Fix FindMKL.cmake script for sequential threading library \PR{2840} \PR{2952}
- Various internal refactoring \PR{2839} \PR{2861} \PR{2864} \PR{2873} \PR{2890} \PR{2891} \PR{2913} \PR{2959}
- Fix OpenCL 2.0 builtin function name conflict \PR{2851}
- Fix error caused when releasing memory with multiple devices \PR{2867}
- Fix missing set stacktrace symbol from unified API \PR{2915}
- Fix zero padding issue in convolve2NN \PR{2820}
- Fixed bugs in ReduceByKey \PR{2957}

Contributions
-------------
Special thanks to our contributors:
[Corentin Schreiber](https://github.com/cschreib-ibex)
[Jacob Kahn](https://github.com/jacobkahn)
[Paul Jurczak](https://github.com/pauljurczak)
[Christoph Junghans](https://github.com/junghans)

v3.7.1
======

Improvements
------------

- Improve mtx download for test data \PR{2742}
- Documentation improvements \PR{2754} \PR{2792} \PR{2797}
- Remove verbose messages in older CMake versions \PR{2773}
- Reduce binary size with the use of nvrtc  \PR{2790}
- Use texture memory to load LUT in orb and fast \PR{2791}
- Add missing print function for f16 \PR{2784}
- Add checks for f16 support in the CUDA backend \PR{2784}
- Create a thrust policy to intercept tmp buffer allocations \PR{2806}

Fixes
-----

- Fix segfault on exit when ArrayFire is not initialized in the main thread
- Fix support for CMake 3.5.1 \PR{2771} \PR{2772} \PR{2760}
- Fix evalMultiple if the input array sizes aren't the same \PR{2766}
- Fix error when AF_BACKEND_DEFAULT is passed directly to backend \PR{2769}
- Workaround name collision with AMD OpenCL implementation \PR{2802}
- Fix on-exit errors with the unified backend \PR{2769}
- Fix check for f16 compatibility in OpenCL \PR{2773}
- Fix matmul on Intel OpenCL when passing same array as input \PR{2774}
- Fix CPU OpenCL blas batching \PR{2774}
- Fix memory pressure in the default memory manager \PR{2801}

Contributions
-------------
Special thanks to our contributors:
[padentomasello](https://github.com/padentomasello)
[glavaux2](https://github.com/glavaux2)

v3.7.0
======

Major Updates
-------------

- Added the ability to customize the memory manager(Thanks jacobkahn and flashlight) \PR{2461}
- Added 16-bit floating point support for several functions \PR{2413} \PR{2587} \PR{2585} \PR{2587} \PR{2583}
- Added sumByKey, productByKey, minByKey, maxByKey, allTrueByKey, anyTrueByKey, countByKey \PR{2254}
- Added confidence connected components \PR{2748}
- Added neural network based convolution and gradient functions \PR{2359}
- Added a padding function \PR{2682}
- Added pinverse for pseudo inverse \PR{2279}
- Added support for uniform ranges in approx1 and approx2 functions. \PR{2297}
- Added support to write to preallocated arrays for some functions \PR{2599} \PR{2481} \PR{2328} \PR{2327}
- Added meanvar function \PR{2258}
- Add support for sparse-sparse arithmetic support
- Added rsqrt function for reciprocal square root
- Added a lower level af_gemm function for general matrix multiplication \PR{2481}
- Added a function to set the cuBLAS math mode for the CUDA backend \PR{2584}
- Separate debug symbols into separate files \PR{2535}
- Print stacktraces on errors \PR{2632}
- Support move constructor for af::array \PR{2595}
- Expose events in the public API \PR{2461}
- Add setAxesLabelFormat to format labels on graphs \PR{2495}

Improvements
------------

- Better error messages for systems with driver or device incompatibilities \PR{2678} \PR{2448}
- Optimized unified backend function calls
- Optimized anisotropic smoothing \PR{2713}
- Optimized canny filter for CUDA and OpenCL
- Better MKL search script
- Better logging of different submodules in ArrayFire \PR{2670} \PR{2669}
- Improve documentation \PR{2665} \PR{2620} \PR{2615} \PR{2639} \PR{2628} \PR{2633} \PR{2622} \PR{2617} \PR{2558} \PR{2326} \PR{2515}
- Optimized af::array assignment \PR{2575}
- Update the k-means example to display the result \PR{2521}


Fixes
-----

- Fix multi-config generators
- Fix access errors in canny
- Fix segfault in the unified backend if no backends are available
- Fix access errors in scan-by-key
- Fix sobel operator
- Fix an issue with the random number generator and s16
- Fix issue with boolean product reduction
- Fix array_proxy move constructor
- Fix convolve3 launch configuration
- Fix an issue where the fft function modified the input array \PR{2520}

Contributions
-------------
Special thanks to our contributors:
[Jacob Khan](https://github.com/jacobkahn)
[William Tambellini](https://github.com/WilliamTambellini)
[Alexey Kuleshevich](https://github.com/lehins)
[Richard Barnes](https://github.com/r-barnes)
[Gaika](https://github.com/gaika)
[ShalokShalom](https://github.com/ShalokShalom)


v3.6.4
======

Bug Fixes
---------
- Address a JIT performance regression due to moving kernel arguments to shared memory \PR{2501}
- Fix the default parameter for setAxisTitle \PR{2491}

v3.6.3
======

Improvements
------------
- Graphics are now a runtime dependency instead of a link time dependency \PR{2365}
- Reduce the CUDA backend binary size using runtime compilation of kernels \PR{2437}
- Improved batched matrix multiplication on the CPU backend by using Intel MKL's
  `cblas_Xgemm_batched`\PR{2206}
- Print JIT kernels to disk or stream using the `AF_JIT_KERNEL_TRACE`
  environment variable \PR{2404}
- `void*` pointers are now allowed as arguments to `af::array::write()` \PR{2367}
- Slightly improve the efficiency of JITed tile operations \PR{2472}
- Make the random number generation on the CPU backend to be consistent with
  CUDA and OpenCL \PR{2435}
- Handled very large JIT tree generations \PR{2484} \PR{2487}

Bug Fixes
---------
- Fixed `af::array::array_proxy` move assignment operator \PR{2479}
- Fixed input array dimensions validation in svdInplace() \PR{2331}
- Fixed the typedef declaration for window resource handle \PR{2357}.
- Increase compatibility with GCC 8 \PR{2379}
- Fixed `af::write` tests \PR{2380}
- Fixed a bug in broadcast step of 1D exclusive scan \PR{2366}
- Fixed OpenGL related build errors on OSX \PR{2382}
- Fixed multiple array evaluation. Performance improvement. \PR{2384}
- Fixed buffer overflow and expected output of kNN SSD small test \PR{2445}
- Fixed MKL linking order to enable threaded BLAS \PR{2444}
- Added validations for forge module plugin availability before calling
  resource cleanup \PR{2443}
- Improve compatibility on MSVC toolchain(_MSC_VER > 1914) with the CUDA
  backend \PR{2443}
- Fixed BLAS gemm func generators for newest MSVC 19 on VS 2017 \PR{2464}
- Fix errors on exits when using the cuda backend with unified \PR{2470}

Documentation
-------------
- Updated svdInplace() documentation following a bugfix \PR{2331}
- Fixed a typo in matrix multiplication documentation \PR{2358}
- Fixed a code snippet demostrating C-API use \PR{2406}
- Updated hamming matcher implementation limitation \PR{2434}
- Added illustration for the rotate function \PR{2453}

Misc
----
- Use cudaMemcpyAsync instead of cudaMemcpy throughout the codebase \PR{2362}
- Display a more informative error message if CUDA driver is incomptible
  \PR{2421} \PR{2448}
- Changed forge resource managemenet to use smart pointers \PR{2452}
- Deprecated intl and uintl typedefs in API \PR{2360}
- Enabled graphics by default for all builds starting with v3.6.3 \PR{2365}
- Fixed several warnings \PR{2344} \PR{2356} \PR{2361}
- Refactored initArray() calls to use createEmptyArray(). initArray() is for
  internal use only by Array class. \PR{2361}
- Refactored `void*` memory allocations to use unsigned char type \PR{2459}
- Replaced deprecated MKL API with in-house implementations for sparse
  to sparse/dense conversions \PR{2312}
- Reorganized and fixed some internal backend API \PR{2356}
- Updated compilation order of cuda files to speed up compile time \PR{2368}
- Removed conditional graphics support builds after enabling runtime
  loading of graphics dependencies \PR{2365}
- Marked graphics dependencies as optional in CPack RPM config \PR{2365}
- Refactored a sparse arithmetic backend API \PR{2379}
- Fixed const correctness of `af_device_array` API \PR{2396}
- Update Forge to v1.0.4 \PR{2466}
- Manage Forge resources from the DeviceManager class \PR{2381}
- Fixed non-mkl & non-batch blas upstream call arguments \PR{2401}
- Link MKL with OpenMP instead of TBB by default
- use clang-format to format source code

Contributions
-------------
Special thanks to our contributors:
[Alessandro Bessi](https://github.com/alessandrobessi)
[zhihaoy](https://github.com/zhihaoy)
[Jacob Khan](https://github.com/jacobkahn)
[William Tambellini](https://github.com/WilliamTambellini)

v3.6.2
======

Features
--------
- Added support for batching on the `cond` argument in select() \PR{2243}
- Added support for broadcasting batched matmul() \PR{2315}
- Added support for multiple nearest neighbors in nearestNeighbour() \PR{2280}
- Added support for clamp-to-edge padding as an `af_border_type` option \PR{2333}

Improvements
------------
- Improved performance of morphological operations \PR{2238}
- Fixed linking errors when compiling without Freeimage/Graphics \PR{2248}
- Improved the usage of ArrayFire as a CMake subproject \PR{2290}
- Enabled configuration of custom library path for loading dynamic backend
  libraries \PR{2302}

Bug Fixes
---------
- Fixed LAPACK definitions and linking errors \PR{2239}
- Fixed overflow in dim4::ndims() \PR{2289}
- Fixed pow() precision for integral types \PR{2305}
- Fixed issues with tile() with a large repeat dimension \PR{2307}
- Fixed svd() sub-array output on OpenCL \PR{2279}
- Fixed grid-based indexing calculation in histogram() \PR{2230}
- Fixed bug in indexing when used after reorder \PR{2311}
- Fixed errors when exiting on Windows when using
  [CLBlast](https://github.com/CNugteren/CLBlast) \PR{2222}
- Fixed fallthrough error in medfilt1 \PR{2349}

Documentation
-------------
- Improved unwrap() documentation \PR{2301}
- Improved wrap() documentation \PR{2320}
- Improved accum() documentation \PR{2298}
- Improved tile() documentation \PR{2293}
- Clarified approx1() and approx2() indexing in documentation \PR{2287}
- Updated examples of [select()](@ref data_func_select) in detailed documentation
  \PR{2277}
- Updated lookup() examples \PR{2288}
- Updated set operations' documentation \PR{2299}

Misc
----
- `af*` libraries and dependencies directory changed to `lib64` \PR{2186}
- Added new arrayfire ASSERT utility functions \PR{2249} \PR{2256} \PR{2257} \PR{2263}
- Improved error messages in JIT \PR{2309}

Contributions
-------------
Special thanks to our contributors: [Jacob Kahn](https://github.com/jacobkahn),
[Vardan Akopian](https://github.com/vakopian)

v3.6.1
======

Improvements
------------
- FreeImage is now a run-time dependency [#2164]
- Reduced binary size by setting the symbol visibility to hidden [#2168]
- Add memory manager logging using the AF_TRACE=mem environment variable [#2169]
- Improved CPU Anisotropic Diffusion performance [#2174]
- Perform normalization after FFT for improved accuracy [#2185][#2192]
- Updated CLBlast to v1.4.0 [#2178]
- Added additional validation when using af::seq for indexing [#2153]
- Perform checks for unsupported cards by the CUDA implementation [#2182]

Bug Fixes
---------
- Fixed region when all pixels were the foreground or background [#2152]
- Fixed several memory leaks [#2202][#2201][#2180][#2179][#2177][#2175]
- Fixed bug in setDevice which didn't allow you to select the last device [#2189]
- Fixed bug in min/max where the first element of the array was a NaN value [#2155]
- Fixed window cell indexing for graphics [#2207]

v3.6.0
======

The source code with submodules can be downloaded directly from the following link:
http://arrayfire.com/arrayfire_source/arrayfire-full-3.6.0.tar.bz2

Major Updates
-------------

- Added the `topk()` function
  [Documentation](http://arrayfire.org/docs/group__stat__func__topk.htm).
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/2061)</sup>
- Added batched matrix multiply support.
  <sup>[2](https://github.com/arrayfire/arrayfire/pull/1898)</sup>
  <sup>[3](https://github.com/arrayfire/arrayfire/pull/2059)</sup>
- Added anisotropic diffusion, `anisotropicDiffusion()`.
  [Documentation](http://arrayfire.org/docs/group__image__func__anisotropic__diffusion.htm)
  <sup>[4](https://github.com/arrayfire/arrayfire/pull/1850)</sup>.

Features
--------

- Added support for batched matrix multiply.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1898)</sup>
  <sup>[2](https://github.com/arrayfire/arrayfire/pull/2059)</sup>
- New anisotropic diffusion function, `anisotropicDiffusion()`.
  [Documentation](http://arrayfire.org/docs/group__image__func__anisotropic__diffusion.htm)
  <sup>[3](https://github.com/arrayfire/arrayfire/pull/1850)</sup>.
- New `topk()` function, which returns the top k elements along a given
  dimension of the input.
  [Documentation](http://arrayfire.org/docs/group__stat__func__topk.htm).
  <sup>[4](https://github.com/arrayfire/arrayfire/pull/2061)</sup>
- New gradient diffusion
  [example](https://github.com/arrayfire/arrayfire/blob/master/examples/image_processing/gradient_diffusion.cpp).

Improvements
------------

- JITted `select()` and `shift()` functions for CUDA and OpenCL backends.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/2047)</sup>
- Significant CMake improvements.
  <sup>[2](https://github.com/arrayfire/arrayfire/pull/1861)</sup>
  <sup>[3](https://github.com/arrayfire/arrayfire/pull/2070)</sup>
  <sup>[4](https://github.com/arrayfire/arrayfire/pull/2018)</sup>
- Improved the quality of the random number generator, thanks to Ralf Stubner.
  <sup>[5](https://github.com/arrayfire/arrayfire/pull/2122)</sup>
- Modified `af_colormap` struct to match forge's definition.
  <sup>[6](https://github.com/arrayfire/arrayfire/pull/2082)</sup>
- Improved Black Scholes example.
  <sup>[7](https://github.com/arrayfire/arrayfire/pull/2079)</sup>
- Using CPack to generate installers.
  <sup>[8](https://github.com/arrayfire/arrayfire/pull/1861)</sup>
- Refactored
  [black_scholes_options](https://github.com/arrayfire/arrayfire/blob/master/examples/financial/black_scholes_options.cpp)
  example to use built-in `af::erfc` function for cumulative normal
  distribution.<sup>[9](https://github.com/arrayfire/arrayfire/pull/2079)</sup>.
- Reduced the scope of mutexes in memory manager
  <sup>[10](https://github.com/arrayfire/arrayfire/pull/2125)</sup>
- Official installers do not require the CUDA toolkit to be installed
- Significant CMake improvements have been made. Using CPack to generate
  installers. <sup>[11](https://github.com/arrayfire/arrayfire/pull/1861)</sup>
  <sup>[12](https://github.com/arrayfire/arrayfire/pull/2070)</sup>
  <sup>[13](https://github.com/arrayfire/arrayfire/pull/2018)</sup>
- Corrected assert function calls in select() tests.
  <sup>[14](https://github.com/arrayfire/arrayfire/pull/2058)</sup>

Bug fixes
-----------

- Fixed `shfl_down()` warnings with CUDA 9.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/2040)</sup>
- Disabled CUDA JIT debug flags on ARM
  architecture.<sup>[2](https://github.com/arrayfire/arrayfire/pull/2037)</sup>
- Fixed CLBLast install lib dir for linux platform where `lib` directory has
  arch(64) suffix.<sup>[3](https://github.com/arrayfire/arrayfire/pull/2094)</sup>
- Fixed assert condition in 3d morph opencl
  kernel.<sup>[4](https://github.com/arrayfire/arrayfire/pull/2033)</sup>
- Fix JIT errors with large non-linear
  kernels<sup>[5](https://github.com/arrayfire/arrayfire/pull/2127)</sup>
- Fix bug in CPU jit after moddims was called
  <sup>[5](https://github.com/arrayfire/arrayfire/pull/2127)</sup>
- Fixed deadlock caused by calls to from the worker thread
  <sup>[6](https://github.com/arrayfire/arrayfire/pull/2124)</sup>

Documentation
-------------

- Fixed variable name typo in `vectorization.md`.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/2032)</sup>
- Fixed `AF_API_VERSION` value in Doxygen config file.
  <sup>[2](https://github.com/arrayfire/arrayfire/pull/2053)</sup>

Known issues
------------

- Several OpenCL tests failing on OSX:
  - `canny_opencl, fft_opencl, gen_assign_opencl, homography_opencl,
    reduce_opencl, scan_by_key_opencl, solve_dense_opencl,
    sparse_arith_opencl, sparse_convert_opencl, where_opencl`

Community contributions
-----------------------

Special thanks to our contributors:
[Adrien F. Vincent](https://github.com/afvincent), [Cedric
Nugteren](https://github.com/CNugteren),
[Felix](https://github.com/fzimmermann89), [Filip
Matzner](https://github.com/FloopCZ),
[HoneyPatouceul](https://github.com/HoneyPatouceul), [Patrick
Lavin](https://github.com/plavin), [Ralf Stubner](https://github.com/rstub),
[William Tambellini](https://github.com/WilliamTambellini)


v3.5.1
======

The source code with submodules can be downloaded directly from the following
link: http://arrayfire.com/arrayfire_source/arrayfire-full-3.5.1.tar.bz2

Installer CUDA Version: 8.0 (Required) Installer OpenCL Version: 1.2 (Minimum)

Improvements
------------
- Relaxed `af::unwrap()` function's arguments.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1853)</sup>
- Changed behavior of af::array::allocated() to specify memory allocated.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1877)</sup>
- Removed restriction on the number of bins for `af::histogram()` on CUDA and
  OpenCL kernels. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1895)</sup>


Performance
-----------

- Improved JIT performance.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1864)</sup>
- Improved CPU element-wise operation performance.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1890)</sup>
- Improved regions performance using texture objects. <sup>
  [1](https://github.com/arrayfire/arrayfire/pull/1903)</sup>


Bug fixes
---------
- Fixed overflow issues in mean.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1849)</sup>
- Fixed memory leak when chaining indexing operations.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1879)</sup>
- Fixed bug in array assignment when using an empty array to index.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1897)</sup>
- Fixed bug with `af::matmul()` which occured when its RHS argument was an
  indexed vector.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1883)</sup>
- Fixed bug deadlock bug when sparse array was used with a JIT Array.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1889)</sup>
- Fixed pixel tests for FAST kernels.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1891)</sup>
- Fixed `af::replace` so that it is now copy-on-write.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1892)</sup>
- Fixed launch configuration issues in CUDA JIT.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1893)</sup>
- Fixed segfaults and "Pure Virtual Call" error warnings when exiting on
  Windows. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1899)
  [2](https://github.com/arrayfire/arrayfire/pull/1924)</sup>
- Workaround for `clEnqueueReadBuffer` bug on OSX.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1888)</sup>

Build
-----

- Fixed issues when compiling with GCC 7.1.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1872)</sup>
  <sup>[2](https://github.com/arrayfire/arrayfire/pull/1876)</sup>
- Eliminated unnecessary Boost dependency from CPU and CUDA backends.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1857)</sup>

Misc
----

- Updated support links to point to Slack instead of Gitter.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1905)</sup>



v3.5.0
==============

Major Updates
-------------

* ArrayFire now supports threaded applications.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1706)</sup>
* Added Canny edge detector.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1743)</sup>
* Added Sparse-Dense arithmetic operations.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1696)</sup>

Features
--------

* ArrayFire Threading
  * \ref af::array can be read by multiple threads
  * All ArrayFire functions can be executed concurrently by multiple threads
  * Threads can operate on different devices to simplify Muli-device workloads
* New Canny edge detector function, \ref af::canny().
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1743)</sup>
  * Can automatically calculate high threshold with `AF_CANNY_THRESHOLD_AUTO_OTSU`
  * Supports both L1 and L2 Norms to calculate gradients
* New tuned OpenCL BLAS backend,
  [CLBlast](https://github.com/arrayfire/arrayfire/pull/1727).

Improvements
------------

* Converted CUDA JIT to use
  [NVRTC](http://docs.nvidia.com/cuda/nvrtc/index.html) instead of
  [NVVM](http://docs.nvidia.com/cuda/nvvm-ir-spec/index.html).
* Performance improvements in \ref af::reorder().
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1766)</sup>
* Performance improvements in \ref af::array::scalar<T>().
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1809)</sup>
* Improved unified backend performance.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1770)</sup>
* ArrayFire now depends on Forge
  v1.0. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1800)</sup>
* Can now specify the FFT plan cache size using the
  \ref af::setFFTPlanCacheSize() function.
* Get the number of physical bytes allocated by the memory manager
  \ref af_get_allocated_bytes(). <sup>[1](https://github.com/arrayfire/arrayfire/pull/1630)</sup>
* \ref af::dot() can now return a scalar value to the
  host. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1628)</sup>

Bug Fixes
---------

* Fixed improper release of default Mersenne random
  engine. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1716)</sup>
* Fixed \ref af::randu() and \ref af::randn() ranges for floating point
  types. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1784)</sup>
* Fixed assignment bug in CPU
  backend. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1765)</sup>
* Fixed complex (`c32`,`c64`) multiplication in OpenCL convolution
  kernels. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1816)</sup>
* Fixed inconsistent behavior with \ref af::replace() and \ref
  af_replace_scalar(). <sup>[1](https://github.com/arrayfire/arrayfire/pull/1773)</sup>
* Fixed memory leak in \ref
  af_fir(). <sup>[1](https://github.com/arrayfire/arrayfire/pull/1765)</sup>
* Fixed memory leaks in \ref af_cast for sparse arrays.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1826)</sup>
* Fixing correctness of \ref af_pow for complex numbers by using Cartesian
  form. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1765)</sup>
* Corrected \ref af::select() with indexing in CUDA and OpenCL
  backends. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1731)</sup>
* Workaround for VS2015 compiler ternary
  bug. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1771)</sup>
* Fixed memory corruption in
  `cuda::findPlan()`. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1793)</sup>
* Argument checks in \ref af_create_sparse_array avoids inputs of type
  int64. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1747)</sup>
* Fixed issue with indexing an array with a step size != 1. <sup>[1](https://github.com/arrayfire/arrayfire/issues/1846)</sup>

Build fixes
-----------

* On OSX, utilize new GLFW package from the brew package
  manager. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1720)</sup>
  <sup>[2](https://github.com/arrayfire/arrayfire/pull/1775)</sup>
* Fixed CUDA PTX names generated by CMake
  v3.7. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1689)</sup>
* Support `gcc` > 5.x for
  CUDA. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1708)</sup>

Examples
--------

* New genetic algorithm example.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1695)</sup>

Documentation
-------------

* Updated `README.md` to improve readability and
  formatting. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1726)</sup>
* Updated `README.md` to mention Julia and Nim
  wrappers. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1714)</sup>
* Improved installation instructions -
  `docs/pages/install.md`. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1740)</sup>

Miscellaneous
-------------

* A few improvements for ROCm
  support. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1710)</sup>
* Removed CUDA 6.5 support.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1687)</sup>

Known issues
------------

* Windows
  * The Windows NVIDIA driver version `37x.xx` contains a bug which causes
    `fftconvolve_opencl` to fail. Upgrade or downgrade to a different version of
    the driver to avoid this failure.
  * The following tests fail on Windows with NVIDIA hardware:
    `threading_cuda`,`qr_dense_opencl`, `solve_dense_opencl`.
* macOS
  * The Accelerate framework, used by the CPU backend on macOS, leverages Intel
    graphics cards (Iris) when there are no discrete GPUs available. This OpenCL
    implementation is known to give incorrect results on the following tests:
    `lu_dense_{cpu,opencl}`, `solve_dense_{cpu,opencl}`,
    `inverse_dense_{cpu,opencl}`.
  * Certain tests intermittently fail on macOS with NVIDIA GPUs apparently due
    to inconsistent driver behavior: `fft_large_cuda` and `svd_dense_cuda`.
  * The following tests are currently failing on macOS with AMD GPUs:
    `cholesky_dense_opencl` and `scan_by_key_opencl`.


v3.4.2
==============

Deprecation Announcement
------------------------

This release supports CUDA 6.5 and higher. The next ArrayFire relase will
support CUDA 7.0 and higher, dropping support for CUDA 6.5. Reasons for no
longer supporting CUDA 6.5 include:

* CUDA 7.0 NVCC supports the C++11 standard (whereas CUDA 6.5 does not), which
  is used by ArrayFire's CPU and OpenCL backends.
* Very few ArrayFire users still use CUDA 6.5.

As a result, the older Jetson TK1 / Tegra K1 will no longer be supported in
the next ArrayFire release. The newer Jetson TX1 / Tegra X1 will continue to
have full capability with ArrayFire.

Docker
------
* [ArrayFire has been Dockerized](https://github.com/arrayfire/arrayfire-docker).

Improvements
------------
* Implemented sparse storage format conversions between \ref AF_STORAGE_CSR
  and \ref AF_STORAGE_COO.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1642)</sup>
  * Directly convert between \ref AF_STORAGE_COO <--> \ref AF_STORAGE_CSR
    using the af::sparseConvertTo() function.
  * af::sparseConvertTo() now also supports converting to dense.
* Added cast support for [sparse arrays](\ref sparse_func).
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1653)</sup>
  * Casting only changes the values array and the type. The row and column
    index arrays are not changed.
* Reintroduced automated computation of chart axes limits for graphics functions.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1639)</sup>
  * The axes limits will always be the minimum/maximum of the current and new
    limit.
  * The user can still set limits from API calls. If the user sets a limit
    from the API call, then the automatic limit setting will be disabled.
* Using `boost::scoped_array` instead of `boost::scoped_ptr` when managing
  array resources.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1637)</sup>
* Internal performance improvements to getInfo() by using `const` references
  to avoid unnecessary copying of `ArrayInfo` objects.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1665)</sup>
* Added support for scalar af::array inputs for af::convolve() and
  [set functions](\ref set_mat).
  <sup>[1](https://github.com/arrayfire/arrayfire/issues/1660)</sup>
  <sup>[2](https://github.com/arrayfire/arrayfire/issues/1675)</sup>
  <sup>[3](https://github.com/arrayfire/arrayfire/pull/1668)</sup>
* Performance fixes in af::fftConvolve() kernels.
  <sup>[1](https://github.com/arrayfire/arrayfire/issues/1679)</sup>
  <sup>[2](https://github.com/arrayfire/arrayfire/pull/1680)</sup>

Build
-----
* Support for Visual Studio 2015 compilation.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1632)</sup>
  <sup>[2](https://github.com/arrayfire/arrayfire/pull/1640)</sup>
* Fixed `FindCBLAS.cmake` when PkgConfig is used.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1657)</sup>

Bug fixes
---------
* Fixes to JIT when tree is large.
  <sup>[1](https://github.com/arrayfire/arrayfire/issues/1646)</sup>
  <sup>[2](https://github.com/arrayfire/arrayfire/pull/1638)</sup>
* Fixed indexing bug when converting dense to sparse af::array as \ref
  AF_STORAGE_COO.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1642)</sup>
* Fixed af::bilateral() OpenCL kernel compilation on OS X.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1638)</sup>
* Fixed memory leak in af::regions() (CPU) and af::rgb2ycbcr().
  <sup>[1](https://github.com/arrayfire/arrayfire/issues/1664)</sup>
  <sup>[2](https://github.com/arrayfire/arrayfire/issues/1664)</sup>
  <sup>[3](https://github.com/arrayfire/arrayfire/pull/1666)</sup>

Installers
----------
* Major OS X installer fixes.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1629)</sup>
  * Fixed installation scripts.
  * Fixed installation symlinks for libraries.
* Windows installer now ships with more pre-built examples.

Examples
--------
* Added af::choleskyInPlace() calls to `cholesky.cpp` example.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1671)</sup>

Documentation
-------------
* Added `u8` as supported data type in `getting_started.md`.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1661)</sup>
* Fixed typos.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1652)</sup>

CUDA 8 on OSX
-------------
* [CUDA 8.0.55](https://developer.nvidia.com/cuda-toolkit) supports Xcode 8.
  <sup>[1](https://github.com/arrayfire/arrayfire/issues/1664)</sup>

Known Issues
------------
* Known failures with CUDA 6.5. These include all functions that use
  sorting. As a result, sparse storage format conversion between \ref
  AF_STORAGE_COO and \ref AF_STORAGE_CSR has been disabled for CUDA 6.5.

v3.4.1
==============

Installers
----------
* Installers for Linux, OS X and Windows
  * CUDA backend now uses [CUDA 8.0](https://developer.nvidia.com/cuda-toolkit).
  * Uses [Intel MKL 2017](https://software.intel.com/en-us/intel-mkl).
  * CUDA Compute 2.x (Fermi) is no longer compiled into the library.
* Installer for OS X
  * The libraries shipping in the OS X Installer are now compiled with Apple
    Clang v7.3.1 (previously v6.1.0).
  * The OS X version used is 10.11.6 (previously 10.10.5).
* Installer for Jetson TX1 / Tegra X1
  * Requires [JetPack for L4T 2.3](https://developer.nvidia.com/embedded/jetpack)
    (containing Linux for Tegra r24.2 for TX1).
  * CUDA backend now uses [CUDA 8.0](https://developer.nvidia.com/cuda-toolkit) 64-bit.
  * Using CUDA's cusolver instead of CPU fallback.
  * Uses OpenBLAS for CPU BLAS.
  * All ArrayFire libraries are now 64-bit.

Improvements
------------
* Add [sparse array](\ref sparse_func) support to \ref af::eval().
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1598)</sup>
* Add OpenCL-CPU fallback support for sparse \ref af::matmul() when running on
  a unified memory device. Uses MKL Sparse BLAS.
* When using CUDA libdevice, pick the correct compute version based on device.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1612)</sup>
* OpenCL FFT now also supports prime factors 7, 11 and 13.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1383)</sup>
  <sup>[2](https://github.com/arrayfire/arrayfire/pull/1619)</sup>

Bug Fixes
---------
* Allow CUDA libdevice to be detected from custom directory.
* Fix `aarch64` detection on Jetson TX1 64-bit OS.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1593)</sup>
* Add missing definition of `af_set_fft_plan_cache_size` in unified backend.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1591)</sup>
* Fix intial values for \ref af::min() and \ref af::max() operations.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1594)</sup>
  <sup>[2](https://github.com/arrayfire/arrayfire/pull/1595)</sup>
* Fix distance calculation in \ref af::nearestNeighbour for CUDA and OpenCL backend.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1596)</sup>
  <sup>[2](https://github.com/arrayfire/arrayfire/pull/1595)</sup>
* Fix OpenCL bug where scalars where are passed incorrectly to compile options.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1595)</sup>
* Fix bug in \ref af::Window::surface() with respect to dimensions and ranges.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1604)</sup>
* Fix possible double free corruption in \ref af_assign_seq().
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1605)</sup>
* Add missing eval for key in \ref af::scanByKey in CPU backend.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1605)</sup>
* Fixed creation of sparse values array using \ref AF_STORAGE_COO.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1620)</sup>
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1621)</sup>

Examples
--------
* Add a [Conjugate Gradient solver example](\ref benchmarks/cg.cpp)
  to demonstrate sparse and dense matrix operations.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1599)</sup>

CUDA Backend
------------
* When using [CUDA 8.0](https://developer.nvidia.com/cuda-toolkit),
  compute 2.x are no longer in default compute list.
  * This follows [CUDA 8.0](https://developer.nvidia.com/cuda-toolkit)
    deprecating computes 2.x.
  * Default computes for CUDA 8.0 will be 30, 50, 60.
* When using CUDA pre-8.0, the default selection remains 20, 30, 50.
* CUDA backend now uses `-arch=sm_30` for PTX compilation as default.
  * Unless compute 2.0 is enabled.

Known Issues
------------
* \ref af::lu() on CPU is known to give incorrect results when built run on
  OS X 10.11 or 10.12 and compiled with Accelerate Framework.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1617)</sup>
  * Since the OS X Installer libraries uses MKL rather than Accelerate
    Framework, this issue does not affect those libraries.


v3.4.0
==============

Major Updates
-------------
* [Sparse Matrix and BLAS](\ref sparse_func). <sup>[1](https://github.com/arrayfire/arrayfire/issues/821)
  [2](https://github.com/arrayfire/arrayfire/pull/1319)</sup>
* Faster JIT for CUDA and OpenCL. <sup>[1](https://github.com/arrayfire/arrayfire/issues/1472)
  [2](https://github.com/arrayfire/arrayfire/pull/1462)</sup>
* Support for [random number generator engines](\ref af::randomEngine).
  <sup>[1](https://github.com/arrayfire/arrayfire/issues/868)
  [2](https://github.com/arrayfire/arrayfire/pull/1551)</sup>
* Improvements to graphics. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1555)
  [2](https://github.com/arrayfire/arrayfire/pull/1566)</sup>

Features
----------
* **[Sparse Matrix and BLAS](\ref sparse_func)** <sup>[1](https://github.com/arrayfire/arrayfire/issues/821)
[2](https://github.com/arrayfire/arrayfire/pull/1319)</sup>
  * Support for [CSR](\ref AF_STORAGE_CSR) and [COO](\ref AF_STORAGE_COO)
    [storage types](\ref af_storage).
  * Sparse-Dense Matrix Multiplication and Matrix-Vector Multiplication as a
    part of af::matmul() using \ref AF_STORAGE_CSR format for sparse.
  * Conversion to and from [dense](\ref AF_STORAGE_DENSE) matrix to [CSR](\ref AF_STORAGE_CSR)
    and [COO](\ref AF_STORAGE_COO) [storage types](\ref af_storage).
* **Faster JIT** <sup>[1](https://github.com/arrayfire/arrayfire/issues/1472)
  [2](https://github.com/arrayfire/arrayfire/pull/1462)</sup>
  * Performance improvements for CUDA and OpenCL JIT functions.
  * Support for evaluating multiple outputs in a single kernel. See af::array::eval() for more.
* **[Random Number Generation](\ref af::randomEngine)**
  <sup>[1](https://github.com/arrayfire/arrayfire/issues/868)
  [2](https://github.com/arrayfire/arrayfire/pull/1551)</sup>
  * af::randomEngine(): A random engine class to handle setting the [type](af_random_type) and seed
    for random number generator engines.
  * Supported engine types are (\ref af_random_engine_type):
    * [Philox](http://www.thesalmons.org/john/random123/)
    * [Threefry](http://www.thesalmons.org/john/random123/)
    * [Mersenne Twister](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MTGP/)
* **Graphics** <sup>[1](https://github.com/arrayfire/arrayfire/pull/1555)
  [2](https://github.com/arrayfire/arrayfire/pull/1566)</sup>
  * Using [Forge v0.9.0](https://github.com/arrayfire/forge/releases/tag/v0.9.0)
  * [Vector Field](\ref af::Window::vectorField) plotting functionality.
    <sup>[1](https://github.com/arrayfire/arrayfire/pull/1566)</sup>
  * Removed [GLEW](http://glew.sourceforge.net/) and replaced with [glbinding](https://github.com/cginternals/glbinding).
    * Removed usage of GLEW after support for MX (multithreaded) was dropped in v2.0.
      <sup>[1](https://github.com/arrayfire/arrayfire/issues/1540)</sup>
  * Multiple overlays on the same window are now possible.
    * Overlays support for same type of object (2D/3D)
    * Supported by af::Window::plot, af::Window::hist, af::Window::surface,
      af::Window::vectorField.
  * New API to set axes limits for graphs.
    * Draw calls do not automatically compute the limits. This is now under user control.
    * af::Window::setAxesLimits can be used to set axes limits automatically or manually.
    * af::Window::setAxesTitles can be used to set axes titles.
  * New API for plot and scatter:
    * af::Window::plot() and af::Window::scatter() now can handle 2D and 3D and determine appropriate order.
    * af_draw_plot_nd()
    * af_draw_plot_2d()
    * af_draw_plot_3d()
    * af_draw_scatter_nd()
    * af_draw_scatter_2d()
    * af_draw_scatter_3d()
* **New [interpolation methods](\ref af_interp_type)**
<sup>[1](https://github.com/arrayfire/arrayfire/issues/1562)</sup>
  * Applies to
    * \ref af::resize()
    * \ref af::transform()
    * \ref af::approx1()
    * \ref af::approx2()
* **Support for [complex mathematical functions](\ref mathfunc_mat)**
  <sup>[1](https://github.com/arrayfire/arrayfire/issues/1507)</sup>
  * Add complex support for \ref trig_mat, \ref af::sqrt(), \ref af::log().
* **af::medfilt1(): Median filter for 1-d signals** <sup>[1](https://github.com/arrayfire/arrayfire/pull/1479)</sup>
* <b>Generalized scan functions: \ref scan_func_scan and \ref scan_func_scanbykey</b>
  * Now supports inclusive or exclusive scans
  * Supports binary operations defined by \ref af_binary_op.
  <sup>[1](https://github.com/arrayfire/arrayfire/issues/388)</sup>
* **[Image Moments](\ref moments_mat) functions**
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1453)</sup>
* <b>Add af::getSizeOf() function for \ref af_dtype</b>
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1404)</sup>
* <b>Explicitly extantiate \ref af::array::device() for `void *</b>
  <sup>[1](https://github.com/arrayfire/arrayfire/issues/1503)</sup>

Bug Fixes
--------------
* Fixes to edge-cases in \ref morph_mat. <sup>[1](https://github.com/arrayfire/arrayfire/issues/1564)</sup>
* Makes JIT tree size consistent between devices. <sup>[1](https://github.com/arrayfire/arrayfire/issues/1457)</sup>
* Delegate higher-dimension in \ref convolve_mat to correct dimensions. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1445)</sup>
* Indexing fixes with C++11. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1426) [2](https://github.com/arrayfire/arrayfire/pull/1426)</sup>
* Handle empty arrays as inputs in various functions. <sup>[1](https://github.com/arrayfire/arrayfire/issues/799)</sup>
* Fix bug when single element input to af::median. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1423)</sup>
* Fix bug in calculation of time from af::timeit(). <sup>[1](https://github.com/arrayfire/arrayfire/pull/1414)</sup>
* Fix bug in floating point numbers in af::seq. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1404)</sup>
* Fixes for OpenCL graphics interop on NVIDIA devices.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1408/commits/e1f16e6)</sup>
* Fix bug when compiling large kernels for AMD devices.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1465)</sup>
* Fix bug in af::bilateral when shared memory is over the limit.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1478)</sup>
* Fix bug in kernel header compilation tool `bin2cpp`.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1544)</sup>
* Fix inital values for \ref morph_mat functions.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1547)</sup>
* Fix bugs in af::homography() CPU and OpenCL kernels.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1584)</sup>
* Fix bug in CPU TNJ.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1587)</sup>


Improvements
------------
* CUDA 8 and compute 6.x(Pascal) support, current installer ships with CUDA 7.5. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1432) [2](https://github.com/arrayfire/arrayfire/pull/1487) [3](https://github.com/arrayfire/arrayfire/pull/1539)</sup>
* User controlled FFT plan caching. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1448)</sup>
* CUDA performance improvements for \ref image_func_wrap, \ref image_func_unwrap and \ref approx_mat.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1411)</sup>
* Fallback for CUDA-OpenGL interop when no devices does not support OpenGL.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1415)</sup>
* Additional forms of batching with the \ref transform_func_transform functions.
  [New behavior defined here](https://github.com/arrayfire/arrayfire/pull/1412).
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1412)</sup>
* Update to OpenCL2 headers. <sup>[1](https://github.com/arrayfire/arrayfire/issues/1344)</sup>
* Support for integration with external OpenCL contexts. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1140)</sup>
* Performance improvements to interal copy in CPU Backend.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1440)</sup>
* Performance improvements to af::select and af::replace CUDA kernels.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1587)</sup>
* Enable OpenCL-CPU offload by default for devices with Unified Host Memory.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1521)</sup>
  * To disable, use the environment variable `AF_OPENCL_CPU_OFFLOAD=0`.

Build
------
* Compilation speedups. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1526)</sup>
* Build fixes with MKL. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1526)</sup>
* Error message when CMake CUDA Compute Detection fails. <sup>[1](https://github.com/arrayfire/arrayfire/issues/1535)</sup>
* Several CMake build issues with Xcode generator fixed.
  <sup>[1](https://github.com/arrayfire/arrayfire/pull/1493) [2](https://github.com/arrayfire/arrayfire/pull/1499)</sup>
* Fix multiple OpenCL definitions at link time. <sup>[1](https://github.com/arrayfire/arrayfire/issues/1429)</sup>
* Fix lapacke detection in CMake. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1423)</sup>
* Update build tags of
  * [clBLAS](https://github.com/clMathLibraries/clBLAS)
  * [clFFT](https://github.com/clMathLibraries/clFFT)
  * [Boost.Compute](https://github.com/boostorg/compute)
  * [Forge](https://github.com/arrayfire/forge)
  * [glbinding](https://github.com/cginternals/glbinding)
* Fix builds with GCC 6.1.1 and GCC 5.3.0. <sup>[1](https://github.com/arrayfire/arrayfire/pull/1409)</sup>

Installers
----------
* All installers now ship with ArrayFire libraries build with MKL 2016.
* All installers now ship with Forge development files and examples included.
* CUDA Compute 2.0 has been removed from the installers. Please contact us
  directly if you have a special need.

Examples
-------------
* Added [example simulating gravity](\ref graphics/field.cpp) for
  demonstration of vector field.
* Improvements to \ref financial/black_scholes_options.cpp example.
* Improvements to \ref graphics/gravity_sim.cpp example.
* Fix graphics examples to use af::Window::setAxesLimits and
  af::Window::setAxesTitles functions.

Documentation & Licensing
-------------------------
* [ArrayFire copyright and trademark policy](http://arrayfire.com/trademark-policy)
* Fixed grammar in license.
* Add license information for glbinding.
* Remove license infomation for GLEW.
* Random123 now applies to all backends.
* Random number functions are now under \ref random_mat.

Deprecations
------------
The following functions have been deprecated and may be modified or removed
permanently from future versions of ArrayFire.
* \ref af::Window::plot3(): Use \ref af::Window::plot instead.
* \ref af_draw_plot(): Use \ref af_draw_plot_nd or \ref af_draw_plot_2d instead.
* \ref af_draw_plot3(): Use \ref af_draw_plot_nd or \ref af_draw_plot_3d instead.
* \ref af::Window::scatter3(): Use \ref af::Window::scatter instead.
* \ref af_draw_scatter(): Use \ref af_draw_scatter_nd or \ref af_draw_scatter_2d instead.
* \ref af_draw_scatter3(): Use \ref af_draw_scatter_nd or \ref af_draw_scatter_3d instead.

Known Issues
-------------
Certain CUDA functions are known to be broken on Tegra K1. The following ArrayFire tests are currently failing:
* assign_cuda
* harris_cuda
* homography_cuda
* median_cuda
* orb_cudasort_cuda
* sort_by_key_cuda
* sort_index_cuda


v3.3.2
==============

Improvements
------------
* Family of [Sort](\ref sort_mat) functions now support
  [higher order dimensions](https://github.com/arrayfire/arrayfire/pull/1373).
* Improved performance of batched sort on dim 0 for all [Sort](\ref sort_mat) functions.
* [Median](\ref stat_func_median) now also supports higher order dimensions.

Bug Fixes
--------------

* Fixes to [error handling](https://github.com/arrayfire/arrayfire/issues/1352) in C++ API for binary functions.
* Fixes to [external OpenCL context management](https://github.com/arrayfire/arrayfire/issues/1350).
* Fixes to [JPEG_GREYSCALE](https://github.com/arrayfire/arrayfire/issues/1360) for FreeImage versions <= 3.154.
* Fixed for [non-float inputs](https://github.com/arrayfire/arrayfire/issues/1386) to \ref af::rgb2gray().

Build
------
* [Disable CPU Async](https://github.com/arrayfire/arrayfire/issues/1378) when building with GCC < 4.8.4.
* Add option to [disable CPUID](https://github.com/arrayfire/arrayfire/issues/1369) from CMake.
* More verbose message when [CUDA Compute Detection fails](https://github.com/arrayfire/arrayfire/issues/1362).
* Print message to use [CUDA library stub](https://github.com/arrayfire/arrayfire/issues/1363)
  from CUDA Toolkit if CUDA Library is not found from default paths.
* [Build Fixes](https://github.com/arrayfire/arrayfire/pull/1385) on Windows.
  * For compiling tests our of source.
  * For compiling ArrayFire with static MKL.
* [Exclude <sys/sysctl.h>](https://github.com/arrayfire/arrayfire/pull/1368) when building on GNU Hurd.
* Add [manual CMake options](https://github.com/arrayfire/arrayfire/pull/1389) to build DEB and RPM packages.

Documentation
-------------
* Fixed documentation for \ref af::replace().
* Fixed images in [Using on OSX](\ref using_on_osx) page.

Installer
---------
* Linux x64 installers will now be compiled with GCC 4.9.2.
* OSX installer gives better error messages on brew failures and
  now includes link to [Fixing OS X Installer Failures] (https://github.com/arrayfire/arrayfire/wiki/Fixing-Common-OS-X-Installer-Failures)
  for brew installation failures.

v3.3.1
==============

Bug Fixes
--------------

* Fixes to \ref af::array::device()
    * CPU Backend: [evaluate arrays](https://github.com/arrayfire/arrayfire/issues/1316)
      before returning pointer with asynchronous calls in CPU backend.
    * OpenCL Backend: [fix segfaults](https://github.com/arrayfire/arrayfire/issues/1324)
      when requested for device pointers on empty arrays.
* Fixed \ref af::operator%() from using [rem to mod](https://github.com/arrayfire/arrayfire/issues/1318).
* Fixed [array destruction](https://github.com/arrayfire/arrayfire/issues/1321)
  when backends are switched in Unified API.
* Fixed [indexing](https://github.com/arrayfire/arrayfire/issues/1331) after
  \ref af::moddims() is called.
* Fixes FFT calls for CUDA and OpenCL backends when used on
  [multiple devices](https://github.com/arrayfire/arrayfire/issues/1332).
* Fixed [unresolved external](https://github.com/arrayfire/arrayfire/commit/32965ef)
  for some functions from \ref af::array::array_proxy class.

Build
------
* CMake compiles files in alphabetical order.
* CMake fixes for BLAS and LAPACK on some Linux distributions.

Improvements
------------
* Fixed [OpenCL FFT performance](https://github.com/arrayfire/arrayfire/issues/1323) regression.
* \ref af::array::device() on OpenCL backend [returns](https://github.com/arrayfire/arrayfire/issues/1311)
  `cl_mem` instead of `(void*)cl::Buffer*`.
* In Unified backend, [load versioned libraries](https://github.com/arrayfire/arrayfire/issues/1312)
  at runtime.

Documentation
------
* Reorganized, cleaner README file.
* Replaced non-free lena image in assets with free-to-distribute lena image.

v3.3.0
==============

Major Updates
-------------

* CPU backend supports aysnchronous execution.
* Performance improvements to OpenCL BLAS and FFT functions.
* Improved performance of memory manager.
* Improvements to visualization functions.
* Improved sorted order for OpenCL devices.
* Integration with external OpenCL projects.

Features
----------

* \ref af::getActiveBackend(): Returns the current backend being used.
* [Scatter plot](https://github.com/arrayfire/arrayfire/pull/1116) added to graphics.
* \ref af::transform() now supports perspective transformation matrices.
* \ref af::infoString(): Returns `af::info()` as a string.
* \ref af::printMemInfo(): Print a table showing information about buffer from the memory manager
    * The \ref AF_MEM_INFO macro prints numbers and total sizes of all buffers (requires including af/macros.h)
* \ref af::allocHost(): Allocates memory on host.
* \ref af::freeHost(): Frees host side memory allocated by arrayfire.
* OpenCL functions can now use CPU implementation.
    * Currently limited to Unified Memory devices (CPU and On-board Graphics).
    * Functions: af::matmul() and all [LAPACK](\ref linalg_mat) functions.
    * Takes advantage of optimized libraries such as MKL without doing memory copies.
    * Use the environment variable `AF_OPENCL_CPU_OFFLOAD=1` to take advantage of this feature.
* Functions specific to OpenCL backend.
    * \ref afcl::addDevice(): Adds an external device and context to ArrayFire's device manager.
    * \ref afcl::deleteDevice(): Removes an external device and context from ArrayFire's device manager.
    * \ref afcl::setDevice(): Sets an external device and context from ArrayFire's device manager.
    * \ref afcl::getDeviceType(): Gets the device type of the current device.
    * \ref afcl::getPlatform(): Gets the platform of the current device.
* \ref af::createStridedArray() allows [array creation user-defined strides](https://github.com/arrayfire/arrayfire/issues/1177) and device pointer.
* [Expose functions](https://github.com/arrayfire/arrayfire/issues/1131) that provide information
  about memory layout of Arrays.
    * \ref af::getStrides(): Gets the strides for each dimension of the array.
    * \ref af::getOffset(): Gets the offsets for each dimension of the array.
    * \ref af::getRawPtr(): Gets raw pointer to the location of the array on device.
    * \ref af::isLinear(): Returns true if all elements in the array are contiguous.
    * \ref af::isOwner(): Returns true if the array owns the raw pointer, false if it is a sub-array.
    * \ref af::getStrides(): Gets the strides of the array.
    * \ref af::getStrides(): Gets the strides of the array.
* \ref af::getDeviceId(): Gets the device id on which the array resides.
* \ref af::isImageIOAvailable(): Returns true if ArrayFire was compiled with Freeimage enabled
* \ref af::isLAPACKAvailable(): Returns true if ArrayFire was compiled with LAPACK functions enabled

Bug Fixes
--------------

* Fixed [errors when using 3D / 4D arrays](https://github.com/arrayfire/arrayfire/pull/1251) in select and replace
* Fixed [JIT errors on AMD devices](https://github.com/arrayfire/arrayfire/pull/1238) for OpenCL backend.
* Fixed [imageio bugs](https://github.com/arrayfire/arrayfire/pull/1229) for 16 bit images.
* Fixed [bugs when loading and storing images](https://github.com/arrayfire/arrayfire/pull/1228) natively.
* Fixed [bug in FFT for NVIDIA GPUs](https://github.com/arrayfire/arrayfire/issues/615) when using OpenCL backend.
* Fixed [bug when using external context](https://github.com/arrayfire/arrayfire/pull/1241) with OpenCL backend.
* Fixed [memory leak](https://github.com/arrayfire/arrayfire/issues/1269) in \ref af_median_all().
* Fixed [memory leaks and performance](https://github.com/arrayfire/arrayfire/pull/1274) in graphics functions.
* Fixed [bugs when indexing followed by moddims](https://github.com/arrayfire/arrayfire/issues/1275).
* \ref af_get_revision() now returns actual commit rather than AF_REVISION.
* Fixed [releasing arrays](https://github.com/arrayfire/arrayfire/issues/1282) when using different backends.
* OS X OpenCL: [LAPACK functions](\ref linalg_mat) on CPU devices use OpenCL offload (previously threw errors).
* [Add support for 32-bit integer image types](https://github.com/arrayfire/arrayfire/pull/1287) in Image IO.
* Fixed [set operations for row vectors](https://github.com/arrayfire/arrayfire/issues/1300)
* Fixed [bugs](https://github.com/arrayfire/arrayfire/issues/1243) in \ref af::meanShift() and af::orb().

Improvements
--------------

* Optionally [offload BLAS and LAPACK](https://github.com/arrayfire/arrayfire/pull/1221) functions to CPU implementations to improve performance.
* Performance improvements to the memory manager.
* Error messages are now more detailed.
* Improved sorted order for OpenCL devices.
* JIT heuristics can now be tweaked using environment variables. See
  [Environment Variables](\ref configuring_environment) tutorial.
* Add `BUILD_<BACKEND>` [options to examples and tests](https://github.com/arrayfire/arrayfire/issues/1286)
  to toggle backends when compiling independently.

Examples
----------

* New visualization [example simulating gravity](\ref graphics/gravity_sim.cpp).

Build
----------

* Support for Intel `icc` compiler
* Support to compile with Intel MKL as a BLAS and LAPACK provider
* Tests are now available for building as standalone (like examples)
* Tests can now be built as a single file for each backend
* Better handling of NONFREE build options
* [Searching for GLEW in CMake default paths](https://github.com/arrayfire/arrayfire/pull/1292)
* Fixes for compiling with MKL on OSX.

Installers
----------
* Improvements to OSX Installer
    * CMake config files are now installed with libraries
    * Independent options for installing examples and documentation components

Deprecations
-----------

* `af_lock_device_arr` is now deprecated to be removed in v4.0.0. Use \ref af_lock_array() instead.
* `af_unlock_device_arr` is now deprecated to be removed in v4.0.0. use \ref af_unlock_array() instead.

Documentation
--------------

* Fixes to documentation for \ref af::matchTemplate().
* Improved documentation for deviceInfo.
* Fixes to documentation for \ref af::exp().

Known Issues
------------

* [Solve OpenCL fails on NVIDIA Maxwell devices](https://github.com/arrayfire/arrayfire/issues/1246)
  for f32 and c32 when M > N and K % 4 is 1 or 2.


v3.2.2
==============

Bug Fixes
--------------

* Fixed [memory leak](https://github.com/arrayfire/arrayfire/pull/1145) in
  CUDA Random number generators
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/1157) in
  af::select() and af::replace() tests
* Fixed [exception](https://github.com/arrayfire/arrayfire/issues/1164)
  thrown when printing empty arrays with af::print()
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/1170) in CPU
  random number generation. Changed the generator to
  [mt19937](http://en.cppreference.com/w/cpp/numeric/random)
* Fixed exception handling (internal)
    * [Exceptions](https://github.com/arrayfire/arrayfire/issues/1188)
      now show function, short file name and line number
    * Added [AF_RETURN_ERROR](https://github.com/arrayfire/arrayfire/issues/1186)
      macro to handle returning errors.
    * Removed THROW macro, and renamed AF_THROW_MSG to AF_THROW_ERR.
* Fixed [bug](https://github.com/arrayfire/arrayfire/commit/9459c6)
  in \ref af::identity() that may have affected CUDA Compute 5.2 cards


Build
------
* Added a [MIN_BUILD_TIME](https://github.com/arrayfire/arrayfire/issues/1193)
  option to build with minimum optimization compiler flags resulting in faster
  compile times
* Fixed [issue](https://github.com/arrayfire/arrayfire/issues/1143) in CBLAS
  detection by CMake
* Fixed tests failing for builds without optional components
  [FreeImage](https://github.com/arrayfire/arrayfire/issues/1143) and
  [LAPACK](https://github.com/arrayfire/arrayfire/issues/1167)
* Added a [test](https://github.com/arrayfire/arrayfire/issues/1192)
  for unified backend
* Only [info and backend tests](https://github.com/arrayfire/arrayfire/issues/1192)
  are now built for unified backend
* [Sort tests](https://github.com/arrayfire/arrayfire/issues/1199)
  execution alphabetically
* Fixed compilation flags and errors in tests and examples
* [Moved AF_REVISION and AF_COMPILER_STR](https://github.com/arrayfire/arrayfire/commit/2287c5)
  into src/backend. This is because as revision is updated with every commit,
  entire ArrayFire would have to be rebuilt in the old code.
    * v3.3 will add a af_get_revision() function to get the revision string.
* [Clean up examples](https://github.com/arrayfire/arrayfire/pull/1158)
    * Remove getchar for Windows (this will be handled by the installer)
    * Other miscellaneous code cleanup
    * Fixed bug in [plot3.cpp](\ref graphics/plot3.cpp) example
* [Rename](https://github.com/arrayfire/arrayfire/commit/35f0fc2) clBLAS/clFFT
  external project suffix from external -> ext
* [Add OpenBLAS](https://github.com/arrayfire/arrayfire/pull/1197) as a
  lapack/lapacke alternative

Improvements
------------
* Added \ref AF_MEM_INFO macro to print memory info from ArrayFire's memory
  manager ([cross issue](https://github.com/arrayfire/arrayfire/issues/1172))
* Added [additional paths](https://github.com/arrayfire/arrayfire/issues/1184)
  for searching for `libaf*` for Unified backend on unix-style OS.
    * Note: This still requires dependencies such as forge, CUDA, NVVM etc to be
      in `LD_LIBRARY_PATH` as described in [Unified Backend](\ref unifiedbackend)
* [Create streams](https://github.com/arrayfire/arrayfire/commit/ed0373f)
  for devices only when required in CUDA Backend

Documentation
------
* [Hide scrollbars](https://github.com/arrayfire/arrayfire/commit/9d218a5)
  appearing for pre and code styles
* Fix [documentation](https://github.com/arrayfire/arrayfire/commit/ac09f91) for af::replace
* Add [code sample](https://github.com/arrayfire/arrayfire/commit/4e06483)
  for converting the output of af::getAvailableBackends() into bools
* Minor fixes in documentation

v3.2.1
==============

Bug Fixes
--------------

* Fixed [bug](https://github.com/arrayfire/arrayfire/pull/1136) in homography()
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/1135) in behavior
  of af::array::device()
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/1129) when
  indexing with span along trailing dimension
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/1127) when
  indexing in [GFor](\ref gfor)
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/1122) in CPU
  information fetching
* Fixed compilation [bug](https://github.com/arrayfire/arrayfire/issues/1117)
  in unified backend caused by missing link library
* Add [missing symbol](https://github.com/arrayfire/arrayfire/pull/1114) for
  af_draw_surface()

Build
------
* Tests can now be used as a [standalone project](https://github.com/arrayfire/arrayfire/pull/1120)
    * Tests can now be built using pre-compiled libraries
    * Similar to how the examples are built
* The install target now installs the examples source irrespective of the
  BUILD_EXAMPLES value
    * Examples are not built if BUILD_EXAMPLES is off

Documentation
------
* HTML documentation is now [built and installed](https://github.com/arrayfire/arrayfire/pull/1109)
  in docs/html
* Added documentation for \ref af::seq class
* Updated [Matrix Manipulation](\ref matrixmanipulation) tutorial
* Examples list is now generated by CMake
    * <a href="examples.htm">Examples</a> are now listed as dir/example.cpp
* Removed dummy groups used for indexing documentation (affcted doxygen < 1.8.9)

v3.2.0
=================

Major Updates
-------------

* Added Unified backend
    * Allows switching backends at runtime
    * Read [Unified Backend](\ref unifiedbackend) for more.
* Support for 16-bit integers (\ref s16 and \ref u16)
    * All functions that support 32-bit interger types (\ref s32, \ref u32),
      now also support 16-bit interger types

Function Additions
------------------
* Unified Backend
    * \ref af::setBackend() - Sets a backend as active
    * \ref af::getBackendCount() - Gets the number of backends available for use
    * \ref af::getAvailableBackends() - Returns information about available backends
    * \ref af::getBackendId() - Gets the backend enum for an array

* Vision
    * \ref af::homography() - Homography estimation
    * \ref af::gloh() - GLOH Descriptor for SIFT

* Image Processing
    * \ref af::loadImageNative() - Load an image as native data without modification
    * \ref af::saveImageNative() - Save an image without modifying data or type

* Graphics
    * \ref af::Window::plot3() - 3-dimensional line plot
    * \ref af::Window::surface() - 3-dimensional curve plot

* Indexing
    * \ref af_create_indexers()
    * \ref af_set_array_indexer()
    * \ref af_set_seq_indexer()
    * \ref af_set_seq_param_indexer()
    * \ref af_release_indexers()

* CUDA Backend Specific
    * \ref afcu::setNativeId() - Set the CUDA device with given native id as active
        * ArrayFire uses a modified order for devices. The native id for a
          device can be retreived using `nvidia-smi`

* OpenCL Backend Specific
    * \ref afcl::setDeviceId() - Set the OpenCL device using the `clDeviceId`

Other Improvements
------------------------
* Added \ref c32 and \ref c64 support for \ref af::isNaN(), \ref af::isInf() and \ref af::iszero()
* Added CPU information for `x86` and `x86_64` architectures in CPU backend's \ref af::info()
* Batch support for \ref af::approx1() and \ref af::approx2()
    * Now can be used with gfor as well
* Added \ref s64 and \ref u64 support to:
    * \ref af::sort() (along with sort index and sort by key)
    * \ref af::setUnique(), \ref af::setUnion(), \ref af::setIntersect()
    * \ref af::convolve() and \ref af::fftConvolve()
    * \ref af::histogram() and \ref af::histEqual()
    * \ref af::lookup()
    * \ref af::mean()
* Added \ref AF_MSG macro

Build Improvements
------------------
* Submodules update is now automatically called if not cloned recursively
* [Fixes for compilation](https://github.com/arrayfire/arrayfire/issues/766) on Visual Studio 2015
* Option to use [fallback to CPU LAPACK](https://github.com/arrayfire/arrayfire/pull/1053)
  for linear algebra functions in case of CUDA 6.5 or older versions.

Bug Fixes
--------------
* Fixed [memory leak](https://github.com/arrayfire/arrayfire/pull/1096) in \ref af::susan()
* Fixed [failing test](https://github.com/arrayfire/arrayfire/commit/144a2db)
  in \ref af::lower() and \ref af::upper() for CUDA compute 53
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/1092) in CUDA for indexing out of bounds
* Fixed [dims check](https://github.com/arrayfire/arrayfire/commit/6975da8) in \ref af::iota()
* Fixed [out-of-bounds access](https://github.com/arrayfire/arrayfire/commit/7fc3856) in \ref af::sift()
* Fixed [memory allocation](https://github.com/arrayfire/arrayfire/commit/5e88e4a) in \ref af::fast() OpenCL
* Fixed [memory leak](https://github.com/arrayfire/arrayfire/pull/994) in image I/O functions
* \ref af::dog() now returns float-point type arrays

Documentation Updates
---------------------
* Improved tutorials documentation
    * More detailed Using on [Linux](\ref using_on_linux), [OSX](\ref using_on_osx),
      [Windows](\ref using_on_windows) pages.
* Added return type information for functions that return different type
  arrays

New Examples
------------
* Graphics
    * [Plot3](\ref graphics/plot3.cpp)
    * [Surface](\ref graphics/surface.cpp)
* [Shallow Water Equation](\ref pde/swe.cpp)
* [Basic](\ref unified/basic.cpp) as a Unified backend example

Installers
-----------
* All installers now include the Unified backend and corresponding CMake files
* Visual Studio projects include Unified in the Platform Configurations
* Added installer for Jetson TX1
* SIFT and GLOH do not ship with the installers as SIFT is protected by
  patents that do not allow commercial distribution without licensing.

v3.1.3
==============

Bug Fixes
---------

* Fixed [bugs](https://github.com/arrayfire/arrayfire/issues/1042) in various OpenCL kernels without offset additions
* Remove ARCH_32 and ARCH_64 flags
* Fix [missing symbols](https://github.com/arrayfire/arrayfire/issues/1040) when freeimage is not found
* Use CUDA driver version for Windows
* Improvements to SIFT
* Fixed [memory leak](https://github.com/arrayfire/arrayfire/issues/1045) in median
* Fixes for Windows compilation when not using MKL [#1047](https://github.com/arrayfire/arrayfire/issues/1047)
* Fixed for building without LAPACK

Other
-------

* Documentation: Fixed documentation for select and replace
* Documentation: Fixed documentation for af_isnan

v3.1.2
==============

Bug Fixes
---------

* Fixed [bug](https://github.com/arrayfire/arrayfire/commit/4698f12) in assign that was causing test to fail
* Fixed bug in convolve. Frequency condition now depends on kernel size only
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/1005) in indexed reductions for complex type in OpenCL backend
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/1006) in kernel name generation in ireduce for OpenCL backend
* Fixed non-linear to linear indices in ireduce
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/1011) in reductions for small arrays
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/1010) in histogram for indexed arrays
* Fixed [compiler error](https://github.com/arrayfire/arrayfire/issues/1015) CPUID for non-compliant devices
* Fixed [failing tests](https://github.com/arrayfire/arrayfire/issues/1008) on i386 platforms
* Add missing AFAPI

Other
-------

* Documentation: Added missing examples and other corrections
* Documentation: Fixed warnings in documentation building
* Installers: Send error messages to log file in OSX Installer

v3.1.1
==============

Installers
-----------

* CUDA backend now depends on CUDA 7.5 toolkit
* OpenCL backend now require OpenCL 1.2 or greater

Bug Fixes
--------------

* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/981) in reductions after indexing
* Fixed [bug](https://github.com/arrayfire/arrayfire/issues/976) in indexing when using reverse indices

Build
------

* `cmake` now includes `PKG_CONFIG` in the search path for CBLAS and LAPACKE libraries
* [heston_model.cpp](\ref financial/heston_model.cpp) example now builds with the default ArrayFire cmake files after installation

Other
------

* Fixed bug in [image_editing.cpp](\ref image_processing/image_editing.cpp)

v3.1.0
==============

Function Additions
------------------
* Computer Vision Functions
    * \ref af::nearestNeighbour() - Nearest Neighbour with SAD, SSD and SHD distances
    * \ref af::harris() - Harris Corner Detector
    * \ref af::susan() - Susan Corner Detector
    * \ref af::sift() - Scale Invariant Feature Transform (SIFT)
        * Method and apparatus for identifying scale invariant features"
          "in an image and use of same for locating an object in an image,\" David"
          "G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application"
          "filed March 8, 1999. Asignee: The University of British Columbia. For"
          "further details, contact David Lowe (lowe@cs.ubc.ca) or the"
          "University-Industry Liaison Office of the University of British"
          "Columbia.")
        * SIFT is available for compiling but does not ship with ArrayFire
          hosted installers/pre-built libraries
    * \ref af::dog() -  Difference of Gaussians

* Image Processing Functions
    * \ref ycbcr2rgb() and \ref rgb2ycbcr() - RGB <->YCbCr color space conversion
    * \ref wrap() and \ref unwrap() Wrap and Unwrap
    * \ref sat() - Summed Area Tables
    * \ref loadImageMem() and \ref saveImageMem() - Load and Save images to/from memory
        * \ref af_image_format - Added imageFormat (af_image_format) enum

* Array & Data Handling
    * \ref copy() - Copy
    * array::lock() and array::unlock() - Lock and Unlock
    * \ref select() and \ref replace() - Select and Replace
    * Get array reference count (af_get_data_ref_count)

* Signal Processing
    * \ref fftInPlace() - 1D in place FFT
    * \ref fft2InPlace() - 2D in place FFT
    * \ref fft3InPlace() - 3D in place FFT
    * \ref ifftInPlace() - 1D in place Inverse FFT
    * \ref ifft2InPlace() - 2D in place Inverse FFT
    * \ref ifft3InPlace() - 3D in place Inverse FFT
    * \ref fftR2C() - Real to complex FFT
    * \ref fftC2R() - Complex to Real FFT

* Linear Algebra
    * \ref svd() and \ref svdInPlace() - Singular Value Decomposition

* Other operations
    * \ref sigmoid() - Sigmoid
    * Sum (with option to replace NaN values)
    * Product (with option to replace NaN values)

* Graphics
    * Window::setSize() - Window resizing using Forge API

* Utility
    * Allow users to set print precision (print, af_print_array_gen)
    * \ref saveArray() and \ref readArray() - Stream arrays to binary files
    * \ref toString() - toString function returns the array and data as a string

* CUDA specific functionality
    * \ref getStream() - Returns default CUDA stream ArrayFire uses for the current device
    * \ref getNativeId() - Returns native id of the CUDA device

Improvements
------------
* dot
    * Allow complex inputs with conjugate option
* AF_INTERP_LOWER interpolation
    * For resize, rotate and transform based functions
* 64-bit integer support
    * For reductions, random, iota, range, diff1, diff2, accum, join, shift
      and tile
* convolve
    * Support for non-overlapping batched convolutions
* Complex Arrays
    * Fix binary ops on complex inputs of mixed types
    * Complex type support for exp
* tile
    * Performance improvements by using JIT when possible.
* Add AF_API_VERSION macro
    * Allows disabling of API to maintain consistency with previous versions
* Other Performance Improvements
    * Use reference counting to reduce unnecessary copies
* CPU Backend
    * Device properties for CPU
    * Improved performance when all buffers are indexed linearly
* CUDA Backend
    * Use streams in CUDA (no longer using default stream)
    * Using async cudaMem ops
    * Add 64-bit integer support for JIT functions
    * Performance improvements for CUDA JIT for non-linear 3D and 4D arrays
* OpenCL Backend
    * Improve compilation times for OpenCL backend
    * Performance improvements for non-linear JIT kernels on OpenCL
    * Improved shared memory load/store in many OpenCL kernels (PR 933)
    * Using cl.hpp v1.2.7

Bug Fixes
---------
* Common
    * Fix compatibility of c32/c64 arrays when operating with scalars
    * Fix median for all values of an array
    * Fix double free issue when indexing (30cbbc7)
    * Fix [bug](https://github.com/arrayfire/arrayfire/issues/901) in rank
    * Fix default values for scale throwing exception
    * Fix conjg raising exception on real input
    * Fix bug when using conjugate transpose for vector input
    * Fix issue with const input for array_proxy::get()
* CPU Backend
    * Fix randn generating same sequence for multiple calls
    * Fix setSeed for randu
    * Fix casting to and from complex
    * Check NULL values when allocating memory
    * Fix [offset issue](https://github.com/arrayfire/arrayfire/issues/923) for CPU element-wise operations

New Examples
------------
* Match Template
* Susan
* Heston Model (contributed by Michael Nowotny)

Installer
----------
* Fixed bug in automatic detection of ArrayFire when using with CMake in Windows
* The Linux libraries are now compiled with static version of FreeImage

Known Issues
------------
* OpenBlas can cause issues with QR factorization in CPU backend
* FreeImage older than 3.10 can cause issues with loadImageMem and
  saveImageMem
* OpenCL backend issues on OSX
    * AMD GPUs not supported because of driver issues
    * Intel CPUs not supported
    * Linear algebra functions do not work on Intel GPUs.
* Stability and correctness issues with open source OpenCL implementations such as Beignet, GalliumCompute.

v3.0.2
==============

Bug Fixes
--------------

* Added missing symbols from the compatible API
* Fixed a bug affecting corner rows and elements in \ref af::grad()
* Fixed linear interpolation bugs affecting large images in the following:
    - \ref af::approx1()
    - \ref af::approx2()
    - \ref af::resize()
    - \ref af::rotate()
    - \ref af::scale()
    - \ref af::skew()
    - \ref af::transform()

Documentation
-----------------

* Added missing documentation for \ref af::constant()
* Added missing documentation for `array::scalar()`
* Added supported input types for functions in `arith.h`

v3.0.1
==============

Bug Fixes
--------------

* Fixed header to work in Visual Studio 2015
* Fixed a bug in batched mode for FFT based convolutions
* Fixed graphics issues on OSX
* Fixed various bugs in visualization functions

Other improvements
---------------

* Improved fractal example
* New OSX installer
* Improved Windows installer
  * Default install path has been changed
* Fixed bug in machine learning examples

<br>

v3.0.0
=================

Major Updates
-------------

* ArrayFire is now open source
* Major changes to the visualization library
* Introducing handle based C API
* New backend: CPU fallback available for systems without GPUs
* Dense linear algebra functions available for all backends
* Support for 64 bit integers

Function Additions
------------------
* Data generation functions
    * range()
    * iota()

* Computer Vision Algorithms
    * features()
        * A data structure to hold features
    * fast()
        * FAST feature detector
    * orb()
        * ORB A feature descriptor extractor

* Image Processing
    * convolve1(), convolve2(), convolve3()
        * Specialized versions of convolve() to enable better batch support
    * fftconvolve1(), fftconvolve2(), fftconvolve3()
        * Convolutions in frequency domain to support larger kernel sizes
    * dft(), idft()
        * Unified functions for calling multi dimensional ffts.
    * matchTemplate()
        * Match a kernel in an image
    * sobel()
        * Get sobel gradients of an image
    * rgb2hsv(), hsv2rgb(), rgb2gray(), gray2rgb()
        * Explicit function calls to colorspace conversions
    * erode3d(), dilate3d()
        * Explicit erode and dilate calls for image morphing

* Linear Algebra
    * matmulNT(), matmulTN(), matmulTT()
        * Specialized versions of matmul() for transposed inputs
    * luInPlace(), choleskyInPlace(), qrInPlace()
        * In place factorizations to improve memory requirements
    * solveLU()
        * Specialized solve routines to improve performance
    * OpenCL backend now Linear Algebra functions

* Other functions
    * lookup() - lookup indices from a table
    * batchFunc() - helper function to perform batch operations

* Visualization functions
    * Support for multiple windows
    * window.hist()
        * Visualize the output of the histogram

* C API
    * Removed old pointer based C API
    * Introducing handle base C API
    * Just In Time compilation available in C API
    * C API has feature parity with C++ API
    * bessel functions removed
    * cross product functions removed
    * Kronecker product functions removed

Performance Improvements
------------------------
* Improvements across the board for OpenCL backend

API Changes
---------------------
* `print` is now af_print()
* seq(): The step parameter is now the third input
    * seq(start, step, end) changed to seq(start, end, step)
* gfor(): The iterator now needs to be seq()

Deprecated Function APIs
------------------------
Deprecated APIs are in af/compatible.h

* devicecount() changed to getDeviceCount()
* deviceset() changed to setDevice()
* deviceget() changed to getDevice()
* loadimage() changed to loadImage()
* saveimage() changed to saveImage()
* gaussiankernel() changed to gaussianKernel()
* alltrue() changed to allTrue()
* anytrue() changed to anyTrue()
* setunique() changed to setUnique()
* setunion() changed to setUnion()
* setintersect() changed to setIntersect()
* histequal() changed to histEqual()
* colorspace() changed to colorSpace()
* filter() deprecated. Use convolve1() and convolve2()
* mul() changed to product()
* deviceprop() changed to deviceProp()

Known Issues
----------------------
* OpenCL backend issues on OSX
    * AMD GPUs not supported because of driver issues
    * Intel CPUs not supported
    * Linear algebra functions do not work on Intel GPUs.
* Stability and correctness issues with open source OpenCL implementations such as Beignet, GalliumCompute.
