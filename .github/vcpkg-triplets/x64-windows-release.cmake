# CMakeModules/vcpkg/vcpkg-triplets/x64-windows.cmake for the Windows CI jobs, with release
# builds only: the CI builds RelWithDebInfo or Release, never Debug, so the debug builds of
# the dependencies would only double vcpkg's build time and cache size. ci.sh adds this
# directory to VCPKG_OVERLAY_TRIPLETS, so that a requested commit that predates this file
# builds with it too. Keep the rest in step with x64-windows.cmake.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_PROVIDED_FORTRAN ON)
set(VCPKG_BUILD_TYPE release)

if(PORT MATCHES "freetype")
  set(VCPKG_CRT_LINKAGE static)
  set(VCPKG_LIBRARY_LINKAGE static)
else()
  set(VCPKG_CRT_LINKAGE dynamic)
  set(VCPKG_LIBRARY_LINKAGE dynamic)
endif()
