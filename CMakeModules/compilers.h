
// This is a generated file. Do not edit!

#ifndef AF_COMPILER_DETECTION_H
#define AF_COMPILER_DETECTION_H

#ifdef __cplusplus
# define AF_COMPILER_IS_Comeau 0
# define AF_COMPILER_IS_Intel 0
# define AF_COMPILER_IS_PathScale 0
# define AF_COMPILER_IS_Embarcadero 0
# define AF_COMPILER_IS_Borland 0
# define AF_COMPILER_IS_Watcom 0
# define AF_COMPILER_IS_OpenWatcom 0
# define AF_COMPILER_IS_SunPro 0
# define AF_COMPILER_IS_HP 0
# define AF_COMPILER_IS_Compaq 0
# define AF_COMPILER_IS_zOS 0
# define AF_COMPILER_IS_IBMClang 0
# define AF_COMPILER_IS_XLClang 0
# define AF_COMPILER_IS_XL 0
# define AF_COMPILER_IS_VisualAge 0
# define AF_COMPILER_IS_NVHPC 0
# define AF_COMPILER_IS_PGI 0
# define AF_COMPILER_IS_Cray 0
# define AF_COMPILER_IS_TI 0
# define AF_COMPILER_IS_FujitsuClang 0
# define AF_COMPILER_IS_Fujitsu 0
# define AF_COMPILER_IS_GHS 0
# define AF_COMPILER_IS_Tasking 0
# define AF_COMPILER_IS_SCO 0
# define AF_COMPILER_IS_ARMCC 0
# define AF_COMPILER_IS_AppleClang 0
# define AF_COMPILER_IS_ARMClang 0
# define AF_COMPILER_IS_Clang 0
# define AF_COMPILER_IS_LCC 0
# define AF_COMPILER_IS_GNU 0
# define AF_COMPILER_IS_MSVC 0
# define AF_COMPILER_IS_ADSP 0
# define AF_COMPILER_IS_IAR 0
# define AF_COMPILER_IS_MIPSpro 0

#if defined(__COMO__)
# undef AF_COMPILER_IS_Comeau
# define AF_COMPILER_IS_Comeau 1

#elif defined(__INTEL_COMPILER) || defined(__ICC)
# undef AF_COMPILER_IS_Intel
# define AF_COMPILER_IS_Intel 1

#elif defined(__PATHCC__)
# undef AF_COMPILER_IS_PathScale
# define AF_COMPILER_IS_PathScale 1

#elif defined(__BORLANDC__) && defined(__CODEGEARC_VERSION__)
# undef AF_COMPILER_IS_Embarcadero
# define AF_COMPILER_IS_Embarcadero 1

#elif defined(__BORLANDC__)
# undef AF_COMPILER_IS_Borland
# define AF_COMPILER_IS_Borland 1

#elif defined(__WATCOMC__) && __WATCOMC__ < 1200
# undef AF_COMPILER_IS_Watcom
# define AF_COMPILER_IS_Watcom 1

#elif defined(__WATCOMC__)
# undef AF_COMPILER_IS_OpenWatcom
# define AF_COMPILER_IS_OpenWatcom 1

#elif defined(__SUNPRO_CC)
# undef AF_COMPILER_IS_SunPro
# define AF_COMPILER_IS_SunPro 1

#elif defined(__HP_aCC)
# undef AF_COMPILER_IS_HP
# define AF_COMPILER_IS_HP 1

#elif defined(__DECCXX)
# undef AF_COMPILER_IS_Compaq
# define AF_COMPILER_IS_Compaq 1

#elif defined(__IBMCPP__) && defined(__COMPILER_VER__)
# undef AF_COMPILER_IS_zOS
# define AF_COMPILER_IS_zOS 1

#elif defined(__open_xl__) && defined(__clang__)
# undef AF_COMPILER_IS_IBMClang
# define AF_COMPILER_IS_IBMClang 1

#elif defined(__ibmxl__) && defined(__clang__)
# undef AF_COMPILER_IS_XLClang
# define AF_COMPILER_IS_XLClang 1

#elif defined(__IBMCPP__) && !defined(__COMPILER_VER__) && __IBMCPP__ >= 800
# undef AF_COMPILER_IS_XL
# define AF_COMPILER_IS_XL 1

#elif defined(__IBMCPP__) && !defined(__COMPILER_VER__) && __IBMCPP__ < 800
# undef AF_COMPILER_IS_VisualAge
# define AF_COMPILER_IS_VisualAge 1

#elif defined(__NVCOMPILER)
# undef AF_COMPILER_IS_NVHPC
# define AF_COMPILER_IS_NVHPC 1

#elif defined(__PGI)
# undef AF_COMPILER_IS_PGI
# define AF_COMPILER_IS_PGI 1

#elif defined(_CRAYC)
# undef AF_COMPILER_IS_Cray
# define AF_COMPILER_IS_Cray 1

#elif defined(__TI_COMPILER_VERSION__)
# undef AF_COMPILER_IS_TI
# define AF_COMPILER_IS_TI 1

#elif defined(__CLANG_FUJITSU)
# undef AF_COMPILER_IS_FujitsuClang
# define AF_COMPILER_IS_FujitsuClang 1

#elif defined(__FUJITSU)
# undef AF_COMPILER_IS_Fujitsu
# define AF_COMPILER_IS_Fujitsu 1

#elif defined(__ghs__)
# undef AF_COMPILER_IS_GHS
# define AF_COMPILER_IS_GHS 1

#elif defined(__TASKING__)
# undef AF_COMPILER_IS_Tasking
# define AF_COMPILER_IS_Tasking 1

#elif defined(__SCO_VERSION__)
# undef AF_COMPILER_IS_SCO
# define AF_COMPILER_IS_SCO 1

#elif defined(__ARMCC_VERSION) && !defined(__clang__)
# undef AF_COMPILER_IS_ARMCC
# define AF_COMPILER_IS_ARMCC 1

#elif defined(__clang__) && defined(__apple_build_version__)
# undef AF_COMPILER_IS_AppleClang
# define AF_COMPILER_IS_AppleClang 1

#elif defined(__clang__) && defined(__ARMCOMPILER_VERSION)
# undef AF_COMPILER_IS_ARMClang
# define AF_COMPILER_IS_ARMClang 1

#elif defined(__clang__)
# undef AF_COMPILER_IS_Clang
# define AF_COMPILER_IS_Clang 1

#elif defined(__LCC__) && (defined(__GNUC__) || defined(__GNUG__) || defined(__MCST__))
# undef AF_COMPILER_IS_LCC
# define AF_COMPILER_IS_LCC 1

#elif defined(__GNUC__) || defined(__GNUG__)
# undef AF_COMPILER_IS_GNU
# define AF_COMPILER_IS_GNU 1

#elif defined(_MSC_VER)
# undef AF_COMPILER_IS_MSVC
# define AF_COMPILER_IS_MSVC 1

#elif defined(_ADI_COMPILER)
# undef AF_COMPILER_IS_ADSP
# define AF_COMPILER_IS_ADSP 1

#elif defined(__IAR_SYSTEMS_ICC__) || defined(__IAR_SYSTEMS_ICC)
# undef AF_COMPILER_IS_IAR
# define AF_COMPILER_IS_IAR 1


#endif

#  if AF_COMPILER_IS_AppleClang

#    if !(((__clang_major__ * 100) + __clang_minor__) >= 400)
#      error Unsupported compiler version
#    endif

# define AF_COMPILER_VERSION_MAJOR (__clang_major__)
# define AF_COMPILER_VERSION_MINOR (__clang_minor__)
# define AF_COMPILER_VERSION_PATCH (__clang_patchlevel__)
# if defined(_MSC_VER)
   /* _MSC_VER = VVRR */
#  define AF_SIMULATE_VERSION_MAJOR (_MSC_VER / 100)
#  define AF_SIMULATE_VERSION_MINOR (_MSC_VER % 100)
# endif
# define AF_COMPILER_VERSION_TWEAK (__apple_build_version__)

#    if ((__clang_major__ * 100) + __clang_minor__) >= 400 && __has_feature(cxx_rvalue_references)
#      define AF_COMPILER_CXX_RVALUE_REFERENCES 1
#    else
#      define AF_COMPILER_CXX_RVALUE_REFERENCES 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 400 && __has_feature(cxx_noexcept)
#      define AF_COMPILER_CXX_NOEXCEPT 1
#    else
#      define AF_COMPILER_CXX_NOEXCEPT 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 400 && __has_feature(cxx_variadic_templates)
#      define AF_COMPILER_CXX_VARIADIC_TEMPLATES 1
#    else
#      define AF_COMPILER_CXX_VARIADIC_TEMPLATES 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 400 && __has_feature(cxx_alignas)
#      define AF_COMPILER_CXX_ALIGNAS 1
#    else
#      define AF_COMPILER_CXX_ALIGNAS 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 400 && __has_feature(cxx_static_assert)
#      define AF_COMPILER_CXX_STATIC_ASSERT 1
#    else
#      define AF_COMPILER_CXX_STATIC_ASSERT 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 400 && __has_feature(cxx_generalized_initializers)
#      define AF_COMPILER_CXX_GENERALIZED_INITIALIZERS 1
#    else
#      define AF_COMPILER_CXX_GENERALIZED_INITIALIZERS 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 400 && __has_feature(cxx_relaxed_constexpr)
#      define AF_COMPILER_CXX_RELAXED_CONSTEXPR 1
#    else
#      define AF_COMPILER_CXX_RELAXED_CONSTEXPR 0
#    endif

#  elif AF_COMPILER_IS_Clang

#    if !(((__clang_major__ * 100) + __clang_minor__) >= 301)
#      error Unsupported compiler version
#    endif

# define AF_COMPILER_VERSION_MAJOR (__clang_major__)
# define AF_COMPILER_VERSION_MINOR (__clang_minor__)
# define AF_COMPILER_VERSION_PATCH (__clang_patchlevel__)
# if defined(_MSC_VER)
   /* _MSC_VER = VVRR */
#  define AF_SIMULATE_VERSION_MAJOR (_MSC_VER / 100)
#  define AF_SIMULATE_VERSION_MINOR (_MSC_VER % 100)
# endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 301 && __has_feature(cxx_rvalue_references)
#      define AF_COMPILER_CXX_RVALUE_REFERENCES 1
#    else
#      define AF_COMPILER_CXX_RVALUE_REFERENCES 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 301 && __has_feature(cxx_noexcept)
#      define AF_COMPILER_CXX_NOEXCEPT 1
#    else
#      define AF_COMPILER_CXX_NOEXCEPT 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 301 && __has_feature(cxx_variadic_templates)
#      define AF_COMPILER_CXX_VARIADIC_TEMPLATES 1
#    else
#      define AF_COMPILER_CXX_VARIADIC_TEMPLATES 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 301 && __has_feature(cxx_alignas)
#      define AF_COMPILER_CXX_ALIGNAS 1
#    else
#      define AF_COMPILER_CXX_ALIGNAS 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 301 && __has_feature(cxx_static_assert)
#      define AF_COMPILER_CXX_STATIC_ASSERT 1
#    else
#      define AF_COMPILER_CXX_STATIC_ASSERT 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 301 && __has_feature(cxx_generalized_initializers)
#      define AF_COMPILER_CXX_GENERALIZED_INITIALIZERS 1
#    else
#      define AF_COMPILER_CXX_GENERALIZED_INITIALIZERS 0
#    endif

#    if ((__clang_major__ * 100) + __clang_minor__) >= 301 && __has_feature(cxx_relaxed_constexpr)
#      define AF_COMPILER_CXX_RELAXED_CONSTEXPR 1
#    else
#      define AF_COMPILER_CXX_RELAXED_CONSTEXPR 0
#    endif

#  elif AF_COMPILER_IS_GNU

#    if !((__GNUC__ * 100 + __GNUC_MINOR__) >= 404)
#      error Unsupported compiler version
#    endif

# if defined(__GNUC__)
#  define AF_COMPILER_VERSION_MAJOR (__GNUC__)
# else
#  define AF_COMPILER_VERSION_MAJOR (__GNUG__)
# endif
# if defined(__GNUC_MINOR__)
#  define AF_COMPILER_VERSION_MINOR (__GNUC_MINOR__)
# endif
# if defined(__GNUC_PATCHLEVEL__)
#  define AF_COMPILER_VERSION_PATCH (__GNUC_PATCHLEVEL__)
# endif

#    if (__GNUC__ * 100 + __GNUC_MINOR__) >= 404 && (__cplusplus >= 201103L || (defined(__GXX_EXPERIMENTAL_CXX0X__) && __GXX_EXPERIMENTAL_CXX0X__))
#      define AF_COMPILER_CXX_RVALUE_REFERENCES 1
#    else
#      define AF_COMPILER_CXX_RVALUE_REFERENCES 0
#    endif

#    if (__GNUC__ * 100 + __GNUC_MINOR__) >= 406 && (__cplusplus >= 201103L || (defined(__GXX_EXPERIMENTAL_CXX0X__) && __GXX_EXPERIMENTAL_CXX0X__))
#      define AF_COMPILER_CXX_NOEXCEPT 1
#    else
#      define AF_COMPILER_CXX_NOEXCEPT 0
#    endif

#    if (__GNUC__ * 100 + __GNUC_MINOR__) >= 404 && (__cplusplus >= 201103L || (defined(__GXX_EXPERIMENTAL_CXX0X__) && __GXX_EXPERIMENTAL_CXX0X__))
#      define AF_COMPILER_CXX_VARIADIC_TEMPLATES 1
#    else
#      define AF_COMPILER_CXX_VARIADIC_TEMPLATES 0
#    endif

#    if (__GNUC__ * 100 + __GNUC_MINOR__) >= 408 && __cplusplus >= 201103L
#      define AF_COMPILER_CXX_ALIGNAS 1
#    else
#      define AF_COMPILER_CXX_ALIGNAS 0
#    endif

#    if (__GNUC__ * 100 + __GNUC_MINOR__) >= 404 && (__cplusplus >= 201103L || (defined(__GXX_EXPERIMENTAL_CXX0X__) && __GXX_EXPERIMENTAL_CXX0X__))
#      define AF_COMPILER_CXX_STATIC_ASSERT 1
#    else
#      define AF_COMPILER_CXX_STATIC_ASSERT 0
#    endif

#    if (__GNUC__ * 100 + __GNUC_MINOR__) >= 404 && (__cplusplus >= 201103L || (defined(__GXX_EXPERIMENTAL_CXX0X__) && __GXX_EXPERIMENTAL_CXX0X__))
#      define AF_COMPILER_CXX_GENERALIZED_INITIALIZERS 1
#    else
#      define AF_COMPILER_CXX_GENERALIZED_INITIALIZERS 0
#    endif

#    if (__GNUC__ * 100 + __GNUC_MINOR__) >= 500 && __cplusplus >= 201402L
#      define AF_COMPILER_CXX_RELAXED_CONSTEXPR 1
#    else
#      define AF_COMPILER_CXX_RELAXED_CONSTEXPR 0
#    endif

#  elif AF_COMPILER_IS_Intel

#    if !(__INTEL_COMPILER >= 1210)
#      error Unsupported compiler version
#    endif

  /* __INTEL_COMPILER = VRP prior to 2021, and then VVVV for 2021 and later,
     except that a few beta releases use the old format with V=2021.  */
# if __INTEL_COMPILER < 2021 || __INTEL_COMPILER == 202110 || __INTEL_COMPILER == 202111
#  define AF_COMPILER_VERSION_MAJOR (__INTEL_COMPILER/100)
#  define AF_COMPILER_VERSION_MINOR (__INTEL_COMPILER/10 % 10)
#  if defined(__INTEL_COMPILER_UPDATE)
#   define AF_COMPILER_VERSION_PATCH (__INTEL_COMPILER_UPDATE)
#  else
#   define AF_COMPILER_VERSION_PATCH (__INTEL_COMPILER   % 10)
#  endif
# else
#  define AF_COMPILER_VERSION_MAJOR (__INTEL_COMPILER)
#  define AF_COMPILER_VERSION_MINOR (__INTEL_COMPILER_UPDATE)
   /* The third version component from --version is an update index,
      but no macro is provided for it.  */
#  define AF_COMPILER_VERSION_PATCH (0)
# endif
# if defined(__INTEL_COMPILER_BUILD_DATE)
   /* __INTEL_COMPILER_BUILD_DATE = YYYYMMDD */
#  define AF_COMPILER_VERSION_TWEAK (__INTEL_COMPILER_BUILD_DATE)
# endif
# if defined(_MSC_VER)
   /* _MSC_VER = VVRR */
#  define AF_SIMULATE_VERSION_MAJOR (_MSC_VER / 100)
#  define AF_SIMULATE_VERSION_MINOR (_MSC_VER % 100)
# endif
# if defined(__GNUC__)
#  define AF_SIMULATE_VERSION_MAJOR (__GNUC__)
# elif defined(__GNUG__)
#  define AF_SIMULATE_VERSION_MAJOR (__GNUG__)
# endif
# if defined(__GNUC_MINOR__)
#  define AF_SIMULATE_VERSION_MINOR (__GNUC_MINOR__)
# endif
# if defined(__GNUC_PATCHLEVEL__)
#  define AF_SIMULATE_VERSION_PATCH (__GNUC_PATCHLEVEL__)
# endif

#    if (__cpp_rvalue_references >= 200610 || __INTEL_COMPILER >= 1210) && ((__cplusplus >= 201103L) || defined(__INTEL_CXX11_MODE__) || defined(__GXX_EXPERIMENTAL_CXX0X__))
#      define AF_COMPILER_CXX_RVALUE_REFERENCES 1
#    else
#      define AF_COMPILER_CXX_RVALUE_REFERENCES 0
#    endif

#    if __INTEL_COMPILER >= 1400 && ((__cplusplus >= 201103L) || defined(__INTEL_CXX11_MODE__) || defined(__GXX_EXPERIMENTAL_CXX0X__))
#      define AF_COMPILER_CXX_NOEXCEPT 1
#    else
#      define AF_COMPILER_CXX_NOEXCEPT 0
#    endif

#    if (__cpp_variadic_templates >= 200704 || __INTEL_COMPILER >= 1210) && ((__cplusplus >= 201103L) || defined(__INTEL_CXX11_MODE__) || defined(__GXX_EXPERIMENTAL_CXX0X__))
#      define AF_COMPILER_CXX_VARIADIC_TEMPLATES 1
#    else
#      define AF_COMPILER_CXX_VARIADIC_TEMPLATES 0
#    endif

#    if __INTEL_COMPILER >= 1500 && ((__cplusplus >= 201103L) || defined(__INTEL_CXX11_MODE__) || defined(__GXX_EXPERIMENTAL_CXX0X__))
#      define AF_COMPILER_CXX_ALIGNAS 1
#    else
#      define AF_COMPILER_CXX_ALIGNAS 0
#    endif

#    if (__cpp_static_assert >= 200410 || __INTEL_COMPILER >= 1210) && ((__cplusplus >= 201103L) || defined(__INTEL_CXX11_MODE__) || defined(__GXX_EXPERIMENTAL_CXX0X__))
#      define AF_COMPILER_CXX_STATIC_ASSERT 1
#    else
#      define AF_COMPILER_CXX_STATIC_ASSERT 0
#    endif

#    if __INTEL_COMPILER >= 1400 && ((__cplusplus >= 201103L) || defined(__INTEL_CXX11_MODE__) || defined(__GXX_EXPERIMENTAL_CXX0X__))
#      define AF_COMPILER_CXX_GENERALIZED_INITIALIZERS 1
#    else
#      define AF_COMPILER_CXX_GENERALIZED_INITIALIZERS 0
#    endif

#    if __cpp_constexpr >= 201304 || (__INTEL_COMPILER >= 1700 && ((__cplusplus >= 201300L) || ((__cplusplus == 201103L) && !defined(__INTEL_CXX11_MODE__)) || ((((__INTEL_COMPILER == 1500) && (__INTEL_COMPILER_UPDATE == 1))) && defined(__GXX_EXPERIMENTAL_CXX0X__) && !defined(__INTEL_CXX11_MODE__) ) || (defined(__INTEL_CXX11_MODE__) && defined(__cpp_aggregate_nsdmi)) ) && !defined(_MSC_VER))
#      define AF_COMPILER_CXX_RELAXED_CONSTEXPR 1
#    else
#      define AF_COMPILER_CXX_RELAXED_CONSTEXPR 0
#    endif

#  elif AF_COMPILER_IS_MSVC

#    if !(_MSC_VER >= 1600)
#      error Unsupported compiler version
#    endif

  /* _MSC_VER = VVRR */
# define AF_COMPILER_VERSION_MAJOR (_MSC_VER / 100)
# define AF_COMPILER_VERSION_MINOR (_MSC_VER % 100)
# if defined(_MSC_FULL_VER)
#  if _MSC_VER >= 1400
    /* _MSC_FULL_VER = VVRRPPPPP */
#   define AF_COMPILER_VERSION_PATCH (_MSC_FULL_VER % 100000)
#  else
    /* _MSC_FULL_VER = VVRRPPPP */
#   define AF_COMPILER_VERSION_PATCH (_MSC_FULL_VER % 10000)
#  endif
# endif
# if defined(_MSC_BUILD)
#  define AF_COMPILER_VERSION_TWEAK (_MSC_BUILD)
# endif

#    if _MSC_VER >= 1600
#      define AF_COMPILER_CXX_RVALUE_REFERENCES 1
#    else
#      define AF_COMPILER_CXX_RVALUE_REFERENCES 0
#    endif

#    if _MSC_VER >= 1900
#      define AF_COMPILER_CXX_NOEXCEPT 1
#    else
#      define AF_COMPILER_CXX_NOEXCEPT 0
#    endif

#    if _MSC_VER >= 1800
#      define AF_COMPILER_CXX_VARIADIC_TEMPLATES 1
#    else
#      define AF_COMPILER_CXX_VARIADIC_TEMPLATES 0
#    endif

#    if _MSC_VER >= 1900
#      define AF_COMPILER_CXX_ALIGNAS 1
#    else
#      define AF_COMPILER_CXX_ALIGNAS 0
#    endif

#    if _MSC_VER >= 1600
#      define AF_COMPILER_CXX_STATIC_ASSERT 1
#    else
#      define AF_COMPILER_CXX_STATIC_ASSERT 0
#    endif

#    if _MSC_FULL_VER >= 180030723
#      define AF_COMPILER_CXX_GENERALIZED_INITIALIZERS 1
#    else
#      define AF_COMPILER_CXX_GENERALIZED_INITIALIZERS 0
#    endif

#    if _MSC_VER >= 1911
#      define AF_COMPILER_CXX_RELAXED_CONSTEXPR 1
#    else
#      define AF_COMPILER_CXX_RELAXED_CONSTEXPR 0
#    endif

#  endif

#  if defined(AF_COMPILER_CXX_NOEXCEPT) && AF_COMPILER_CXX_NOEXCEPT
#    define AF_NOEXCEPT noexcept
#    define AF_NOEXCEPT_EXPR(X) noexcept(X)
#  else
#    define AF_NOEXCEPT
#    define AF_NOEXCEPT_EXPR(X)
#  endif


#  if defined(AF_COMPILER_CXX_ALIGNAS) && AF_COMPILER_CXX_ALIGNAS
#    define AF_ALIGNAS(X) alignas(X)
#  elif AF_COMPILER_IS_GNU || AF_COMPILER_IS_Clang || AF_COMPILER_IS_AppleClang
#    define AF_ALIGNAS(X) __attribute__ ((__aligned__(X)))
#  elif AF_COMPILER_IS_MSVC
#    define AF_ALIGNAS(X) __declspec(align(X))
#  else
#    define AF_ALIGNAS(X)
#  endif

#  if defined(AF_COMPILER_CXX_STATIC_ASSERT) && AF_COMPILER_CXX_STATIC_ASSERT
#    define AF_STATIC_ASSERT(X) static_assert(X, #X)
#    define AF_STATIC_ASSERT_MSG(X, MSG) static_assert(X, MSG)
#  else
#    define AF_STATIC_ASSERT_JOIN(X, Y) AF_STATIC_ASSERT_JOIN_IMPL(X, Y)
#    define AF_STATIC_ASSERT_JOIN_IMPL(X, Y) X##Y
template<bool> struct AFStaticAssert;
template<> struct AFStaticAssert<true>{};
#    define AF_STATIC_ASSERT(X) enum { AF_STATIC_ASSERT_JOIN(AFStaticAssertEnum, __LINE__) = sizeof(AFStaticAssert<X>) }
#    define AF_STATIC_ASSERT_MSG(X, MSG) enum { AF_STATIC_ASSERT_JOIN(AFStaticAssertEnum, __LINE__) = sizeof(AFStaticAssert<X>) }
#  endif

#endif

  #if defined(AF_COMPILER_CXX_RELAXED_CONSTEXPR) && AF_COMPILER_CXX_RELAXED_CONSTEXPR
  #define AF_CONSTEXPR constexpr
  #else
  #define AF_CONSTEXPR
  #endif
  #if defined(__cpp_if_constexpr) || __cplusplus >= 201606L
  #define AF_IF_CONSTEXPR if constexpr
  #else
  #define AF_IF_CONSTEXPR if
  #endif


#endif
