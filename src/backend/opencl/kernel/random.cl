/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 *
 ********************************************************/

typedef ulong uint64_t;
typedef uint  uint32_t;

#define PI_VAL 3.1415926535897932384626433832795028841971693993751058209749445923078164
#define R123_STATIC_INLINE inline
#define R123_0x1p_32f (1.f/4294967296.f)
#define R123_0x1p_64 (1./(4294967296.*4294967296.))

#ifdef IS_64
#define SKEIN_KS_PARITY SKEIN_KS_PARITY64
#define RotL RotL_64
#define uint_t uint64_t

#define result(a) (a)*R123_0x1p_64

enum r123_enum_threefry64x2 {
    /*
    // Output from skein_rot_search: (srs64_B64-X1000)
    // Random seed = 1. BlockSize = 128 bits. sampleCnt =  1024. rounds =  8, minHW_or=57
    // Start: Tue Mar  1 10:07:48 2011
    // rMin = 0.136. #0325[*15] [CRC=455A682F. hw_OR=64. cnt=16384. blkSize= 128].format
    */
    R_2_0_0=16,
    R_2_1_0=42,
    R_2_2_0=12,
    R_2_3_0=31,
    R_2_4_0=16,
    R_2_5_0=32,
    R_2_6_0=24,
    R_2_7_0=21
    /* 4 rounds: minHW =  4  [  4  4  4  4 ]
    // 5 rounds: minHW =  8  [  8  8  8  8 ]
    // 6 rounds: minHW = 16  [ 16 16 16 16 ]
    // 7 rounds: minHW = 32  [ 32 32 32 32 ]
    // 8 rounds: minHW = 64  [ 64 64 64 64 ]
    // 9 rounds: minHW = 64  [ 64 64 64 64 ]
    //10 rounds: minHW = 64  [ 64 64 64 64 ]
    //11 rounds: minHW = 64  [ 64 64 64 64 ] */
};
#else
#define SKEIN_KS_PARITY SKEIN_KS_PARITY32
#define RotL RotL_32
#define uint_t uint32_t

#ifdef IS_BOOL
#define result(a) ((a)*R123_0x1p_32f) > 0.5
#else
#define result(a) (a)*R123_0x1p_32f
#endif

enum r123_enum_threefry32x2 {
    /* Output from skein_rot_search (srs2-X5000.out)
    // Random seed = 1. BlockSize = 64 bits. sampleCnt =  1024. rounds =  8, minHW_or=28
    // Start: Tue Jul 12 11:11:33 2011
    // rMin = 0.334. #0206[*07] [CRC=1D9765C0. hw_OR=32. cnt=16384. blkSize=  64].format   */
    R_2_0_0=13,
    R_2_1_0=15,
    R_2_2_0=26,
    R_2_3_0= 6,
    R_2_4_0=17,
    R_2_5_0=29,
    R_2_6_0=16,
    R_2_7_0=24

    /* 4 rounds: minHW =  4  [  4  4  4  4 ]
    // 5 rounds: minHW =  6  [  6  8  6  8 ]
    // 6 rounds: minHW =  9  [  9 12  9 12 ]
    // 7 rounds: minHW = 16  [ 16 24 16 24 ]
    // 8 rounds: minHW = 32  [ 32 32 32 32 ]
    // 9 rounds: minHW = 32  [ 32 32 32 32 ]
    //10 rounds: minHW = 32  [ 32 32 32 32 ]
    //11 rounds: minHW = 32  [ 32 32 32 32 ] */
};
#endif

enum r123_enum_threefry_wcnt {
    WCNT2=2,
    WCNT4=4
};

R123_STATIC_INLINE uint64_t RotL_64(uint64_t x, uint64_t N)
{
    return (x << (N & 63)) | (x >> ((64-N) & 63));
}

R123_STATIC_INLINE uint32_t RotL_32(uint32_t x, uint32_t N)
{
    return (x << (N & 31)) | (x >> ((32-N) & 31));
}

#define SKEIN_MK_64(hi32,lo32)  ((lo32) + (((uint64_t) (hi32)) << 32))
#define SKEIN_KS_PARITY64         SKEIN_MK_64(0x1BD11BDA,0xA9FC1A22)
#define SKEIN_KS_PARITY32         0x1BD11BDA


// http://www.thesalmons.org/john/random123/releases/1.06/docs/structr123_1_1Threefry2x32__R.html#af5be46f8426cfcd86e75327e4b3750b0
#define Nrounds 16

struct r123array2
{
    uint_t v[2];
};

typedef struct r123array2 threefry2_ctr_t;
typedef struct r123array2 threefry2_key_t;
typedef struct r123array2 threefry2_ukey_t;

R123_STATIC_INLINE
threefry2_key_t threefry2keyinit(threefry2_ukey_t uk) { return uk; }

R123_STATIC_INLINE
threefry2_ctr_t threefry2_R(threefry2_ctr_t in, threefry2_key_t k)
{
    threefry2_ctr_t X;
    uint_t ks[2+1];
    int  i; /* avoid size_t to avoid need for stddef.h */
    ks[2] =  SKEIN_KS_PARITY;
    for (i=0;i < 2; i++)
    {
        ks[i] = k.v[i];
        X.v[i]  = in.v[i];
        ks[2] ^= k.v[i];
    }

    /* Insert initial key before round 0 */
    X.v[0] += ks[0]; X.v[1] += ks[1];

    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_0_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_1_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_2_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_3_0); X.v[1] ^= X.v[0];

    /* InjectKey(r=1) */
    X.v[0] += ks[1]; X.v[1] += ks[2];
    X.v[1] += 1;     /* X.v[2-1] += r  */

    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_4_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_5_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_6_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_7_0); X.v[1] ^= X.v[0];

    /* InjectKey(r=2) */
    X.v[0] += ks[2]; X.v[1] += ks[0];
    X.v[1] += 2;

    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_0_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_1_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_2_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_3_0); X.v[1] ^= X.v[0];

    /* InjectKey(r=3) */
    X.v[0] += ks[0]; X.v[1] += ks[1];
    X.v[1] += 3;

    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_4_0); X.v[1] ^= X.v[0];

    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_5_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_6_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_7_0); X.v[1] ^= X.v[0];

    /* InjectKey(r=4) */
    X.v[0] += ks[1]; X.v[1] += ks[2];
    X.v[1] += 4;

#if Nrounds > 16
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_0_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_1_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_2_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_3_0); X.v[1] ^= X.v[0];

    /* InjectKey(r=4) */
    X.v[0] += ks[2]; X.v[1] += ks[0];
    X.v[1] += 5;
#endif

#if Nrounds > 20
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_0_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_1_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_2_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_3_0); X.v[1] ^= X.v[0];

    /* InjectKey(r=3) */
    X.v[0] += ks[0]; X.v[1] += ks[1];
    X.v[1] += 6;
#endif

#if Nrounds > 24
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_4_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_5_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_6_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_7_0); X.v[1] ^= X.v[0];

    /* InjectKey(r=4) */
    X.v[0] += ks[1]; X.v[1] += ks[2];
    X.v[1] += 7;
#endif

#if Nrounds > 28
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_0_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_1_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_2_0); X.v[1] ^= X.v[0];
    X.v[0] += X.v[1]; X.v[1] = RotL(X.v[1],R_2_3_0); X.v[1] ^= X.v[0];

        /* InjectKey(r=4) */
    X.v[0] += ks[2]; X.v[1] += ks[0];
    X.v[1] += 8;
#endif

    return X;
}

#define threefry2(c,k) threefry2_R(c, k)

#ifdef randu
void generate(T *one, T *two, threefry2_ctr_t *c, threefry2_key_t k)
{
    threefry2_ctr_t r = threefry2(*c, k);
    c->v[0] = c->v[0] + 1;

    *one = result(r.v[0]);
    *two = result(r.v[1]);
}
#endif

#ifdef randn
void generate(T *one, T *two, threefry2_ctr_t *c, threefry2_key_t k)
{
    threefry2_ctr_t r = threefry2(*c, k);
    c->v[0] = c->v[0] + 1;

    T u1 = result(r.v[0]);
    T u2 = result(r.v[1]);

#if defined(IS_APPLE) // Because Apple is.. "special"
    T R     = sqrt((T)(-2.0) * log10(u1) * (T)log10_val);
#else
    T R     = sqrt((T)(-2.0) * log(u1));
#endif

    T Theta = 2 * (T)PI_VAL * u2;

    *one = R * sin(Theta);
    *two = R * cos(Theta);
}
#endif

#ifdef randi
void generate(T *one, T *two, threefry2_ctr_t *c, threefry2_key_t k)
{
    threefry2_ctr_t r = threefry2(*c, k);
    c->v[0] = c->v[0] + 1;

    *one = (T)r.v[0];
    *two = (T)r.v[1];
}
#endif

__kernel void random(__global T *output, unsigned numel,
                    unsigned counter, unsigned lo, unsigned hi)
{
    unsigned gid = get_group_id(0);
    unsigned off = get_local_size(0);
    unsigned tid =  off * gid * repeat + get_local_id(0);

    threefry2_key_t k = {{tid, lo}};
    threefry2_ctr_t c = {{counter, hi}};

    T one, two;

    if (gid < get_num_groups(0) - 1) {
        for(int i = 0; i < repeat; i+=2) {
            generate(&one, &two, &c, k);
            output[tid      ] = one;
            output[tid + off] = two;
            tid += 2 * off;
        }
    } else {
        for(int i = 0; i < repeat; i+=2) {
            generate(&one, &two, &c, k);
            if (tid       < numel) output[tid      ] = one;
            if (tid + off < numel) output[tid + off] = two;
            tid += 2 * off;
        }
    }
}
