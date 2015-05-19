/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/***********************************************************************
 * Based on MAGMA library http://icl.cs.utk.edu/magma/
 * Below is the original copyright.
 *
 *   -- MAGMA (version 1.1) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date
 *
 * -- Innovative Computing Laboratory
 * -- Electrical Engineering and Computer Science Department
 * -- University of Tennessee
 * -- (C) Copyright 2009-2013
 *
 * Redistribution  and  use  in  source and binary forms, with or without
 * modification,  are  permitted  provided  that the following conditions
 * are met:
 *
 * * Redistributions  of  source  code  must  retain  the above copyright
 *   notice,  this  list  of  conditions  and  the  following  disclaimer.
 * * Redistributions  in  binary  form must reproduce the above copyright
 *   notice,  this list of conditions and the following disclaimer in the 
 *   documentation  and/or other materials provided with the distribution.
 * * Neither  the  name of the University of Tennessee, Knoxville nor the 
 *   names of its contributors may be used to endorse or promote products
 *   derived from this software without specific prior written permission.
 *
 * THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT 
 * LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
 * THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 * OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **********************************************************************/

#ifndef MAGMA_TYPES_H
#define MAGMA_TYPES_H

#include <stdint.h>
#include <assert.h>
typedef int magma_int_t;
typedef int magma_index_t;

// Define new type that the precision generator will not change (matches PLASMA)
typedef double real_Double_t;

#include <clBLAS.h>

typedef cl_command_queue  magma_queue_t;
typedef cl_event          magma_event_t;
typedef cl_device_id      magma_device_t;

typedef cl_double2 magmaDoubleComplex;
typedef cl_float2  magmaFloatComplex;

#define MAGMA_Z_MAKE(r,i)     doubleComplex(r,i)
#define MAGMA_Z_REAL(a)       (a).s[0]
#define MAGMA_Z_IMAG(a)       (a).s[1]
#define MAGMA_Z_ADD(a, b)     MAGMA_Z_MAKE((a).s[0]+(b).s[0], (a).s[1]+(b).s[1])
#define MAGMA_Z_SUB(a, b)     MAGMA_Z_MAKE((a).s[0]-(b).s[0], (a).s[1]-(b).s[1])
#define MAGMA_Z_DIV(a, b)     ((a)/(b))
#define MAGMA_Z_ABS(a)        magma_cabs(a)
#define MAGMA_Z_ABS1(a)       (fabs((a).s[0]) + fabs((a).s[1]))
#define MAGMA_Z_CNJG(a)       MAGMA_Z_MAKE((a).s[0], -(a).s[1])

#define MAGMA_C_MAKE(r,i)     floatComplex(r,i)
#define MAGMA_C_REAL(a)       (a).s[0]
#define MAGMA_C_IMAG(a)       (a).s[1]
#define MAGMA_C_ADD(a, b)     MAGMA_C_MAKE((a).s[0]+(b).s[0], (a).s[1]+(b).s[1])
#define MAGMA_C_SUB(a, b)     MAGMA_C_MAKE((a).s[0]-(b).s[0], (a).s[1]-(b).s[1])
#define MAGMA_C_DIV(a, b)     ((a)/(b))
#define MAGMA_C_ABS(a)        magma_cabsf(a)
#define MAGMA_C_ABS1(a)       (fabsf((a).s[0]) + fabsf((a).s[1]))
#define MAGMA_C_CNJG(a)       MAGMA_C_MAKE((a).s[0], -(a).s[1])

#define MAGMA_Z_EQUAL(a,b)        (MAGMA_Z_REAL(a)==MAGMA_Z_REAL(b) && MAGMA_Z_IMAG(a)==MAGMA_Z_IMAG(b))
#define MAGMA_Z_NEGATE(a)         MAGMA_Z_MAKE( -MAGMA_Z_REAL(a), -MAGMA_Z_IMAG(a))

#define MAGMA_C_EQUAL(a,b)        (MAGMA_C_REAL(a)==MAGMA_C_REAL(b) && MAGMA_C_IMAG(a)==MAGMA_C_IMAG(b))
#define MAGMA_C_NEGATE(a)         MAGMA_C_MAKE( -MAGMA_C_REAL(a), -MAGMA_C_IMAG(a))

#define MAGMA_D_MAKE(r,i)         (r)
#define MAGMA_D_REAL(x)           (x)
#define MAGMA_D_IMAG(x)           (0.0)
#define MAGMA_D_ADD(a, b)         ((a) + (b))
#define MAGMA_D_SUB(a, b)         ((a) - (b))
#define MAGMA_D_MUL(a, b)         ((a) * (b))
#define MAGMA_D_DIV(a, b)         ((a) / (b))
#define MAGMA_D_ABS(a)            ((a)>0 ? (a) : -(a))
#define MAGMA_D_ABS1(a)           ((a)>0 ? (a) : -(a))
#define MAGMA_D_CNJG(a)           (a)
#define MAGMA_D_EQUAL(a,b)        ((a) == (b))
#define MAGMA_D_NEGATE(a)         (-a)

#define MAGMA_S_MAKE(r,i)         (r)
#define MAGMA_S_REAL(x)           (x)
#define MAGMA_S_IMAG(x)           (0.0)
#define MAGMA_S_ADD(a, b)         ((a) + (b))
#define MAGMA_S_SUB(a, b)         ((a) - (b))
#define MAGMA_S_MUL(a, b)         ((a) * (b))
#define MAGMA_S_DIV(a, b)         ((a) / (b))
#define MAGMA_S_ABS(a)            ((a)>0 ? (a) : -(a))
#define MAGMA_S_ABS1(a)           ((a)>0 ? (a) : -(a))
#define MAGMA_S_CNJG(a)           (a)
#define MAGMA_S_EQUAL(a,b)        ((a) == (b))
#define MAGMA_S_NEGATE(a)         (-a)

#define MAGMA_Z_ZERO              MAGMA_Z_MAKE( 0.0, 0.0)
#define MAGMA_Z_ONE               MAGMA_Z_MAKE( 1.0, 0.0)
#define MAGMA_Z_HALF              MAGMA_Z_MAKE( 0.5, 0.0)
#define MAGMA_Z_NEG_ONE           MAGMA_Z_MAKE(-1.0, 0.0)
#define MAGMA_Z_NEG_HALF          MAGMA_Z_MAKE(-0.5, 0.0)

#define MAGMA_C_ZERO              MAGMA_C_MAKE( 0.0, 0.0)
#define MAGMA_C_ONE               MAGMA_C_MAKE( 1.0, 0.0)
#define MAGMA_C_HALF              MAGMA_C_MAKE( 0.5, 0.0)
#define MAGMA_C_NEG_ONE           MAGMA_C_MAKE(-1.0, 0.0)
#define MAGMA_C_NEG_HALF          MAGMA_C_MAKE(-0.5, 0.0)

#define MAGMA_D_ZERO              ( 0.0)
#define MAGMA_D_ONE               ( 1.0)
#define MAGMA_D_HALF              ( 0.5)
#define MAGMA_D_NEG_ONE           (-1.0)
#define MAGMA_D_NEG_HALF          (-0.5)

#define MAGMA_S_ZERO              ( 0.0)
#define MAGMA_S_ONE               ( 1.0)
#define MAGMA_S_HALF              ( 0.5)
#define MAGMA_S_NEG_ONE           (-1.0)
#define MAGMA_S_NEG_HALF          (-0.5)

#ifndef CBLAS_SADDR
#define CBLAS_SADDR(a)  &(a)
#endif

// OpenCL uses opaque memory references on GPU
typedef cl_mem magma_ptr;
typedef cl_mem magmaInt_ptr;
typedef cl_mem magmaIndex_ptr;
typedef cl_mem magmaFloat_ptr;
typedef cl_mem magmaDouble_ptr;
typedef cl_mem magmaFloatComplex_ptr;
typedef cl_mem magmaDoubleComplex_ptr;

typedef cl_mem magma_const_ptr;
typedef cl_mem magmaInt_const_ptr;
typedef cl_mem magmaIndex_const_ptr;
typedef cl_mem magmaFloat_const_ptr;
typedef cl_mem magmaDouble_const_ptr;
typedef cl_mem magmaFloatComplex_const_ptr;
typedef cl_mem magmaDoubleComplex_const_ptr;


// ========================================
// MAGMA constants

// ----------------------------------------
#define MAGMA_VERSION_MAJOR 1
#define MAGMA_VERSION_MINOR 0
#define MAGMA_VERSION_MICRO 0

// stage is "svn", "beta#", "rc#" (release candidate), or blank ("") for final release
#define MAGMA_VERSION_STAGE "svn"

#define MagmaMaxGPUs 8
#define MagmaMaxSubs 16


// ----------------------------------------
// Return codes
// LAPACK argument errors are < 0 but > MAGMA_ERR.
// MAGMA errors are < MAGMA_ERR.
#define MAGMA_SUCCESS               0
#define MAGMA_ERR                  -100
#define MAGMA_ERR_NOT_INITIALIZED  -101
#define MAGMA_ERR_REINITIALIZED    -102
#define MAGMA_ERR_NOT_SUPPORTED    -103
#define MAGMA_ERR_ILLEGAL_VALUE    -104
#define MAGMA_ERR_NOT_FOUND        -105
#define MAGMA_ERR_ALLOCATION       -106
#define MAGMA_ERR_INTERNAL_LIMIT   -107
#define MAGMA_ERR_UNALLOCATED      -108
#define MAGMA_ERR_FILESYSTEM       -109
#define MAGMA_ERR_UNEXPECTED       -110
#define MAGMA_ERR_SEQUENCE_FLUSHED -111
#define MAGMA_ERR_HOST_ALLOC       -112
#define MAGMA_ERR_DEVICE_ALLOC     -113
#define MAGMA_ERR_CUDASTREAM       -114
#define MAGMA_ERR_INVALID_PTR      -115
#define MAGMA_ERR_UNKNOWN          -116
#define MAGMA_ERR_NOT_IMPLEMENTED  -117


// ----------------------------------------
// parameter constants
// numbering is consistent with CBLAS and PLASMA; see plasma/include/plasma.h
// also with lapack_cwrapper/include/lapack_enum.h
typedef enum {
    MagmaFalse         = 0,
    MagmaTrue          = 1
} magma_bool_t;

typedef enum {
    MagmaRowMajor      = 101,
    MagmaColMajor      = 102
} magma_order_t;

// Magma_ConjTrans is an alias for those rare occasions (zlarfb, zun*, zher*k)
// where we want Magma_ConjTrans to convert to MagmaTrans in precision generation.
typedef enum {
    MagmaNoTrans       = 111,
    MagmaTrans         = 112,
    MagmaConjTrans     = 113,
    Magma_ConjTrans    = MagmaConjTrans
} magma_trans_t;

typedef enum {
    MagmaUpper         = 121,
    MagmaLower         = 122,
    MagmaUpperLower    = 123,
    MagmaFull          = 123   /* lascl, laset */
} magma_uplo_t;

typedef magma_uplo_t magma_type_t;  /* lascl */

typedef enum {
    MagmaNonUnit       = 131,
    MagmaUnit          = 132
} magma_diag_t;

typedef enum {
    MagmaLeft          = 141,
    MagmaRight         = 142,
    MagmaBothSides     = 143   /* trevc */
} magma_side_t;

typedef enum {
    MagmaOneNorm       = 171,  /* lange, lanhe */
    MagmaRealOneNorm   = 172,
    MagmaTwoNorm       = 173,
    MagmaFrobeniusNorm = 174,
    MagmaInfNorm       = 175,
    MagmaRealInfNorm   = 176,
    MagmaMaxNorm       = 177,
    MagmaRealMaxNorm   = 178
} magma_norm_t;

typedef enum {
    MagmaDistUniform   = 201,  /* latms */
    MagmaDistSymmetric = 202,
    MagmaDistNormal    = 203
} magma_dist_t;

typedef enum {
    MagmaHermGeev      = 241,  /* latms */
    MagmaHermPoev      = 242,
    MagmaNonsymPosv    = 243,
    MagmaSymPosv       = 244
} magma_sym_t;

typedef enum {
    MagmaNoPacking     = 291,  /* latms */
    MagmaPackSubdiag   = 292,
    MagmaPackSupdiag   = 293,
    MagmaPackColumn    = 294,
    MagmaPackRow       = 295,
    MagmaPackLowerBand = 296,
    MagmaPackUpeprBand = 297,
    MagmaPackAll       = 298
} magma_pack_t;

typedef enum {
    MagmaNoVec         = 301,  /* geev, syev, gesvd */
    MagmaVec           = 302,  /* geev, syev */
    MagmaIVec          = 303,  /* stedc */
    MagmaAllVec        = 304,  /* gesvd, trevc */
    MagmaSomeVec       = 305,  /* gesvd, trevc */
    MagmaOverwriteVec  = 306,  /* gesvd */
    MagmaBacktransVec  = 307   /* trevc */
} magma_vec_t;

typedef enum {
    MagmaRangeAll      = 311,  /* syevx, etc. */
    MagmaRangeV        = 312,
    MagmaRangeI        = 313
} magma_range_t;

typedef enum {
    MagmaQ             = 322,  /* unmbr, ungbr */
    MagmaP             = 323
} magma_vect_t;

typedef enum {
    MagmaForward       = 391,  /* larfb */
    MagmaBackward      = 392
} magma_direct_t;

typedef enum {
    MagmaColumnwise    = 401,  /* larfb */
    MagmaRowwise       = 402
} magma_storev_t;

// --------------------
// sparse
typedef enum {
    Magma_CSR          = 411,
    Magma_ELLPACK      = 412,
    Magma_ELL          = 413,
    Magma_DENSE        = 414,
    Magma_BCSR         = 415,
    Magma_CSC          = 416,
    Magma_HYB          = 417,
    Magma_COO          = 418,
    Magma_ELLRT        = 419,
    Magma_SELLC        = 420,
    Magma_SELLP        = 421,
    Magma_ELLD         = 422,
    Magma_ELLDD        = 423,
    Magma_CSRD         = 424,
    Magma_CSRL         = 427,
    Magma_CSRU         = 428,
    Magma_CSRCOO       = 429
} magma_storage_t;


typedef enum {
    Magma_CG           = 431,
    Magma_CGMERGE      = 432,
    Magma_GMRES        = 433,
    Magma_BICGSTAB     = 434,
  Magma_BICGSTABMERGE  = 435,
  Magma_BICGSTABMERGE2 = 436,
    Magma_JACOBI       = 437,
    Magma_GS           = 438,
    Magma_ITERREF      = 439,
    Magma_BCSRLU       = 440,
    Magma_PCG          = 441,
    Magma_PGMRES       = 442,
    Magma_PBICGSTAB    = 443,
    Magma_PASTIX       = 444,
    Magma_ILU          = 445,
    Magma_ICC          = 446,
    Magma_AILU         = 447,
    Magma_AICC         = 448,
    Magma_BAITER       = 449,
    Magma_LOBPCG       = 450,
    Magma_NONE         = 451
} magma_solver_type;

typedef enum {
    Magma_CGS          = 461,
    Magma_FUSED_CGS    = 462,
    Magma_MGS          = 463
} magma_ortho_t;

typedef enum {
    Magma_CPU          = 471,
    Magma_DEV          = 472
} magma_location_t;

typedef enum {
    Magma_GENERAL      = 481,
    Magma_SYMMETRIC    = 482
} magma_symmetry_t;

typedef enum {
    Magma_ORDERED      = 491,
    Magma_DIAGFIRST    = 492,
    Magma_UNITY        = 493,
    Magma_VALUE        = 494
} magma_diagorder_t;

typedef enum {
    Magma_DCOMPLEX     = 501,
    Magma_FCOMPLEX     = 502,
    Magma_DOUBLE       = 503,
    Magma_FLOAT        = 504
} magma_precision;

typedef enum {
    Magma_NOSCALE      = 511,
    Magma_UNITROW      = 512,
    Magma_UNITDIAG     = 513
} magma_scale_t;


// When adding constants, remember to do these steps as appropriate:
// 1)  add magma_xxxx_const()  converter below and in control/constants.cpp
// 2a) add to magma2lapack_constants[] in control/constants.cpp
// 2b) update min & max here, which are used to check bounds for magma2lapack_constants[]
// 2c) add lapack_xxxx_const() converter below and in control/constants.cpp
#define Magma2lapack_Min  MagmaFalse     // 0
#define Magma2lapack_Max  MagmaRowwise   // 402


// ----------------------------------------
// string constants for calling Fortran BLAS and LAPACK
// todo: use translators instead? lapack_const( MagmaUpper )
#define MagmaRowMajorStr      "Row"
#define MagmaColMajorStr      "Col"

#define MagmaNoTransStr       "NoTrans"
#define MagmaTransStr         "Trans"
#define MagmaConjTransStr     "ConjTrans"

#define MagmaUpperStr         "Upper"
#define MagmaLowerStr         "Lower"
#define MagmaUpperLowerStr    "Full"
#define MagmaFullStr          "Full"

#define MagmaNonUnitStr       "NonUnit"
#define MagmaUnitStr          "Unit"

#define MagmaLeftStr          "Left"
#define MagmaRightStr         "Right"
#define MagmaBothSidesStr     "Both"

#define MagmaOneNormStr       "1"
#define MagmaTwoNormStr       "2"
#define MagmaFrobeniusNormStr "Fro"
#define MagmaInfNormStr       "Inf"
#define MagmaMaxNormStr       "Max"

#define MagmaForwardStr       "Forward"
#define MagmaBackwardStr      "Backward"

#define MagmaColumnwiseStr    "Columnwise"
#define MagmaRowwiseStr       "Rowwise"

#define MagmaNoVecStr         "NoVec"
#define MagmaVecStr           "Vec"
#define MagmaIVecStr          "IVec"
#define MagmaAllVecStr        "All"
#define MagmaSomeVecStr       "Some"
#define MagmaOverwriteVecStr  "Overwrite"


#ifdef __cplusplus
extern "C" {
#endif

// --------------------
// Convert LAPACK character constants to MAGMA constants.
// This is a one-to-many mapping, requiring multiple translators
// (e.g., "N" can be NoTrans or NonUnit or NoVec).
magma_bool_t   magma_bool_const  ( char lapack_char );
magma_order_t  magma_order_const ( char lapack_char );
magma_trans_t  magma_trans_const ( char lapack_char );
magma_uplo_t   magma_uplo_const  ( char lapack_char );
magma_diag_t   magma_diag_const  ( char lapack_char );
magma_side_t   magma_side_const  ( char lapack_char );
magma_norm_t   magma_norm_const  ( char lapack_char );
magma_dist_t   magma_dist_const  ( char lapack_char );
magma_sym_t    magma_sym_const   ( char lapack_char );
magma_pack_t   magma_pack_const  ( char lapack_char );
magma_vec_t    magma_vec_const   ( char lapack_char );
magma_range_t  magma_range_const ( char lapack_char );
magma_vect_t   magma_vect_const  ( char lapack_char );
magma_direct_t magma_direct_const( char lapack_char );
magma_storev_t magma_storev_const( char lapack_char );


// --------------------
// Convert MAGMA constants to LAPACK(E) constants.
// The generic lapack_const works for all cases, but the specific routines
// (e.g., lapack_trans_const) do better error checking.
const char* lapack_const       ( int            magma_const );
const char* lapack_bool_const  ( magma_bool_t   magma_const );
const char* lapack_order_const ( magma_order_t  magma_const );
const char* lapack_trans_const ( magma_trans_t  magma_const );
const char* lapack_uplo_const  ( magma_uplo_t   magma_const );
const char* lapack_diag_const  ( magma_diag_t   magma_const );
const char* lapack_side_const  ( magma_side_t   magma_const );
const char* lapack_norm_const  ( magma_norm_t   magma_const );
const char* lapack_dist_const  ( magma_dist_t   magma_const );
const char* lapack_sym_const   ( magma_sym_t    magma_const );
const char* lapack_pack_const  ( magma_pack_t   magma_const );
const char* lapack_vec_const   ( magma_vec_t    magma_const );
const char* lapack_range_const ( magma_range_t  magma_const );
const char* lapack_vect_const  ( magma_vect_t   magma_const );
const char* lapack_direct_const( magma_direct_t magma_const );
const char* lapack_storev_const( magma_storev_t magma_const );

static inline char lapacke_const       ( int magma_const            ) { return *lapack_const       ( magma_const ); }
static inline char lapacke_bool_const  ( magma_bool_t   magma_const ) { return *lapack_bool_const  ( magma_const ); }
static inline char lapacke_order_const ( magma_order_t  magma_const ) { return *lapack_order_const ( magma_const ); }
static inline char lapacke_trans_const ( magma_trans_t  magma_const ) { return *lapack_trans_const ( magma_const ); }
static inline char lapacke_uplo_const  ( magma_uplo_t   magma_const ) { return *lapack_uplo_const  ( magma_const ); }
static inline char lapacke_diag_const  ( magma_diag_t   magma_const ) { return *lapack_diag_const  ( magma_const ); }
static inline char lapacke_side_const  ( magma_side_t   magma_const ) { return *lapack_side_const  ( magma_const ); }
static inline char lapacke_norm_const  ( magma_norm_t   magma_const ) { return *lapack_norm_const  ( magma_const ); }
static inline char lapacke_dist_const  ( magma_dist_t   magma_const ) { return *lapack_dist_const  ( magma_const ); }
static inline char lapacke_sym_const   ( magma_sym_t    magma_const ) { return *lapack_sym_const   ( magma_const ); }
static inline char lapacke_pack_const  ( magma_pack_t   magma_const ) { return *lapack_pack_const  ( magma_const ); }
static inline char lapacke_vec_const   ( magma_vec_t    magma_const ) { return *lapack_vec_const   ( magma_const ); }
static inline char lapacke_range_const ( magma_range_t  magma_const ) { return *lapack_range_const ( magma_const ); }
static inline char lapacke_vect_const  ( magma_vect_t   magma_const ) { return *lapack_vect_const  ( magma_const ); }
static inline char lapacke_direct_const( magma_direct_t magma_const ) { return *lapack_direct_const( magma_const ); }
static inline char lapacke_storev_const( magma_storev_t magma_const ) { return *lapack_storev_const( magma_const ); }


// --------------------
// Convert MAGMA constants to clBLAS constants.
#if defined(HAVE_clBLAS)
clblasOrder          clblas_order_const( magma_order_t order );
clblasTranspose      clblas_trans_const( magma_trans_t trans );
clblasUplo           clblas_uplo_const ( magma_uplo_t  uplo  );
clblasDiag           clblas_diag_const ( magma_diag_t  diag  );
clblasSide           clblas_side_const ( magma_side_t  side  );
#endif


// --------------------
// Convert MAGMA constants to CUBLAS constants.
#if defined(CUBLAS_V2_H_)
cublasOperation_t    cublas_trans_const ( magma_trans_t trans );
cublasFillMode_t     cublas_uplo_const  ( magma_uplo_t  uplo  );
cublasDiagType_t     cublas_diag_const  ( magma_diag_t  diag  );
cublasSideMode_t     cublas_side_const  ( magma_side_t  side  );
#endif


// --------------------
// Convert MAGMA constants to CBLAS constants.
#if defined(HAVE_CBLAS)
#include <cblas.h>
enum CBLAS_ORDER     cblas_order_const  ( magma_order_t order );
enum CBLAS_TRANSPOSE cblas_trans_const  ( magma_trans_t trans );
enum CBLAS_UPLO      cblas_uplo_const   ( magma_uplo_t  uplo  );
enum CBLAS_DIAG      cblas_diag_const   ( magma_diag_t  diag  );
enum CBLAS_SIDE      cblas_side_const   ( magma_side_t  side  );
#endif


#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_TYPES_H
