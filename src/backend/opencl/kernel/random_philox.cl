/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 *
 ********************************************************/

/*******************************************************
 * Modified version of Random123 library:
 * https://www.deshawresearch.com/downloads/download_random123.cgi/
 * The original copyright can be seen here:
 *
 * RANDOM123 LICENSE AGREEMENT
 *
 * Copyright 2010-2011, D. E. Shaw Research. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions, and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions, and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * Neither the name of D. E. Shaw Research nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *********************************************************/
typedef uint  uint32_t;

#define PI_VAL 3.1415926535897932384626433832795028841971693993751058209749445923078164
#define m4x32_0                 0xD2511F53u
#define m4x32_1                 0xCD9E8D57u
#define w32_0                   0x9E3779B9u
#define w32_1                   0xBB67AE85u
#define PHILOX_DEFAULT_ROUNDS   10u
#define R                       PHILOX_DEFAULT_ROUNDS
#define UINTMAXFLOAT            4294967296.0f
#define UINTLMAXDOUBLE          4294967296.0*4294967296.0

struct philox_32_4_key_t
{
    uint v[2];
};

struct philox_32_4_ctr_t
{
    uint v[4];
};

void mulhilo(uint a, uint const * const b,
    uint * const hi, uint * const lo)
{
    *hi = mul_hi(a, *b);
    *lo = a*(*b);
}

void philox_round(struct philox_32_4_ctr_t * const ctr,
    struct philox_32_4_key_t const * const key)
{
    uint hi0, lo0, hi1, lo1;
    mulhilo(m4x32_0, &(ctr->v[0]), &hi0, &lo0);
    mulhilo(m4x32_1, &(ctr->v[2]), &hi1, &lo1);
    ctr->v[0] = hi1^(ctr->v[1])^(key->v[0]);
    ctr->v[1] = lo1;
    ctr->v[2] = hi0^(ctr->v[3])^(key->v[1]);
    ctr->v[3] = lo0;
}

void philox_bump(struct philox_32_4_key_t * const key)
{
    key->v[0] += w32_0;
    key->v[1] += w32_1;
}

struct philox_32_4_ctr_t philox(struct philox_32_4_ctr_t ctr,
    struct philox_32_4_key_t key)
{
    struct philox_32_4_key_t const * const keyptr_read_only = (struct philox_32_4_key_t const * const) (&key);
    struct philox_32_4_key_t * const keyptr = (struct philox_32_4_key_t * const) (&key);
    struct philox_32_4_ctr_t * const ctrptr = (struct philox_32_4_ctr_t * const) (&ctr);
#if R > 0
    philox_round(ctrptr, keyptr_read_only);
#endif
#if R > 1
    philox_bump(keyptr); philox_round(ctrptr, keyptr_read_only);
#endif
#if R > 2
    philox_bump(keyptr); philox_round(ctrptr, keyptr_read_only);
#endif
#if R > 3
    philox_bump(keyptr); philox_round(ctrptr, keyptr_read_only);
#endif
#if R > 4
    philox_bump(keyptr); philox_round(ctrptr, keyptr_read_only);
#endif
#if R > 5
    philox_bump(keyptr); philox_round(ctrptr, keyptr_read_only);
#endif
#if R > 6
    philox_bump(keyptr); philox_round(ctrptr, keyptr_read_only);
#endif
#if R > 7
    philox_bump(keyptr); philox_round(ctrptr, keyptr_read_only);
#endif
#if R > 8
    philox_bump(keyptr); philox_round(ctrptr, keyptr_read_only);
#endif
#if R > 9
    philox_bump(keyptr); philox_round(ctrptr, keyptr_read_only);
#endif
#if R > 10
    philox_bump(keyptr); philox_round(ctrptr, keyptr_read_only);
#endif
#if R > 11
    philox_bump(keyptr); philox_round(ctrptr, keyptr_read_only);
#endif
#if R > 12
    philox_bump(keyptr); philox_round(ctrptr, keyptr_read_only);
#endif
#if R > 13
    philox_bump(keyptr); philox_round(ctrptr, keyptr_read_only);
#endif
#if R > 14
    philox_bump(keyptr); philox_round(ctrptr, keyptr_read_only);
#endif
#if R > 15
    philox_bump(keyptr); philox_round(ctrptr, keyptr_read_only);
#endif

return ctr;
}

float normalize_to_float(uint val)
{
    return ((float)(val)/UINTMAXFLOAT);
}

double normalize_to_double(ulong val)
{
    return ((double)(val)/UINTLMAXDOUBLE);
}

#define write_out_4_vals(CONVERT)\
    output[(*tid)]            = CONVERT(r->v[0]);\
    output[(*tid) +   (*off)] = CONVERT(r->v[1]);\
    output[(*tid) + 2*(*off)] = CONVERT(r->v[2]);\
    output[(*tid) + 3*(*off)] = CONVERT(r->v[3]);\
    (*tid) += 4*(*off);\

#define write_out_2_vals(CONVERT)\
    output[(*tid)] = CONVERT((ulong(r->v[0])<<32) | (ulong(r->v[1])));\
    output[(*tid) + (*off)] = CONVERT((ulong(r->v[2])<<32) | (ulong(r->v[3])));\
    (*tid) += 2*(*off);\

#define write_out_8_vals(CONVERT)\
    out[(*tid)]         = CONVERT((r->v[0])&0x00001111);\
    out[(*tid) +   (*off)] = CONVERT((r->v[0])>>4);\
    out[(*tid) + 2*(*off)] = CONVERT((r->v[1])&0x00001111);\
    out[(*tid) + 3*(*off)] = CONVERT((r->v[1])>>4);\
    out[(*tid) + 4*(*off)] = CONVERT((r->v[2])&0x00001111);\
    out[(*tid) + 5*(*off)] = CONVERT((r->v[2])>>4);\
    out[(*tid) + 6*(*off)] = CONVERT((r->v[3])&0x00001111);\
    out[(*tid) + 7*(*off)] = CONVERT((r->v[3])>>4);\
    (*tid) += 8*(*off);\

#define write_out_bool\
    for(uint i = 0; (i < 16); ++i) {\
        out[(*tid)] = ((r->v[i>>2]) & (1 << (i & 3)))? 1:0;\
        (*tid) += (*off);\
    }\

#define write_out_uchar\
    for(uint i = 0; (i < 16); ++i) {\
        out[(*tid)] = (r->v[i>>2] >> ((i & 3) << 1)) & 3;\
        (*tid) += (*off);\
    }\

#define partial_write_out_4_vals(CONVERT)\
    if ((*tid) < (*numel)) {output[(*tid)]            = CONVERT(r->v[0]); *(tid) += (*off);}\
    if ((*tid) < (*numel)) {output[(*tid) +   (*off)] = CONVERT(r->v[1]); *(tid) += (*off);}\
    if ((*tid) < (*numel)) {output[(*tid) + 2*(*off)] = CONVERT(r->v[2]); *(tid) += (*off);}\
    if ((*tid) < (*numel)) {output[(*tid) + 3*(*off)] = CONVERT(r->v[3]); *(tid) += (*off);}

#define partial_write_out_2_vals(CONVERT)\
    if ((*tid) < (*numel)) {output[(*tid)] = CONVERT((ulong(r->v[0])<<32) | (ulong(r->v[1]))); (*tid) += (*off);}\
    if ((*tid) < (*numel)) {output[(*tid) + (*off)] = CONVERT((ulong(r->v[2])<<32) | (ulong(r->v[3]))); (*tid) += (*off);}

#define partial_write_out_8_vals(CONVERT)\
    if ((*tid) < (*numel)) {out[(*tid)]            = CONVERT((r->v[0])&0x00001111); (*tid) += (*off);}\
    if ((*tid) < (*numel)) {out[(*tid) +   (*off)] = CONVERT((r->v[0])>>4); (*tid) += (*off);}\
    if ((*tid) < (*numel)) {out[(*tid) + 2*(*off)] = CONVERT((r->v[1])&0x00001111); (*tid) += (*off);}\
    if ((*tid) < (*numel)) {out[(*tid) + 3*(*off)] = CONVERT((r->v[1])>>4); (*tid) += (*off);}\
    if ((*tid) < (*numel)) {out[(*tid) + 4*(*off)] = CONVERT((r->v[2])&0x00001111); (*tid) += (*off);}\
    if ((*tid) < (*numel)) {out[(*tid) + 5*(*off)] = CONVERT((r->v[2])>>4); (*tid) += (*off);}\
    if ((*tid) < (*numel)) {out[(*tid) + 6*(*off)] = CONVERT((r->v[3])&0x00001111); (*tid) += (*off);}\
    if ((*tid) < (*numel)) {out[(*tid) + 7*(*off)] = CONVERT((r->v[3])>>4); (*tid) += (*off);}

#define partial_write_out_bool\
    for(uint i = 0; (i < 16) && ((*tid) < (*numel)); (*tid) += (*off), ++i) {\
        out[*tid] = ((r->v[i>>2]) & (1 << (i & 3)))? 1:0;\
    }

#define partial_write_out_uchar\
    for(uint i = 0; (i < 16) && ((*tid) < (*numel)); (*tid) += (*off), ++i) {\
        out[*tid] = (r->v[i>>2] >> ((i & 3) << 1)) & 3;\
    }

void generate(__global T *output,
    unsigned * const tid,
    unsigned const * const off,
    struct philox_32_4_ctr_t const * const r)
{
#ifdef inType_float
    write_out_4_vals(normalize_to_float);
#endif
#ifdef inType_double
    write_out_2_vals(normalize_to_double);
#endif
#ifdef inType_int
    write_out_4_vals(int);
#endif
#ifdef inType_intl
    write_out_2_vals(intl);
#endif
#ifdef inType_uint
    write_out_4_vals(uint);
#endif
#ifdef inType_ulong
    write_out_2_vals(ulong);
#endif
#ifdef inType_short
    write_out_8_vals(short);
#endif
#ifdef inType_ushort
    write_out_8_vals(ushort);
#endif
#ifdef inType_char
    write_out_bool;
#endif
#ifdef inType_uchar
    write_out_uchar;
#endif
}

void generate_partial(__global T *output,
    unsigned * const tid,
    unsigned const * const numel,
    unsigned const * const off,
    struct philox_32_4_ctr_t const * const r)
{
#ifdef inType_float
    partial_write_out_4_vals(normalize_to_float);
#endif
#ifdef inType_double
    partial_write_out_2_vals(normalize_to_double);
#endif
#ifdef inType_int
    partial_write_out_4_vals(int);
#endif
#ifdef inType_intl
    partial_write_out_2_vals(intl);
#endif
#ifdef inType_uint
    partial_write_out_4_vals(uint);
#endif
#ifdef inType_ulong
    partial_write_out_2_vals(ulong);
#endif
#ifdef inType_short
    partial_write_out_8_vals(short);
#endif
#ifdef inType_ushort
    partial_write_out_8_vals(ushort);
#endif
#ifdef inType_char
    partial_write_out_bool;
#endif
#ifdef inType_uchar
    partial_write_out_uchar;
#endif
}

__kernel void random_philox(__global T *output, unsigned numel,
                    unsigned counter, unsigned lo, unsigned hi)
{
    unsigned gid = get_group_id(0);
    unsigned off = get_local_size(0);
    unsigned tid =  off * gid * repeat + get_local_id(0);

    uint one, two, three, four;
    struct philox_32_4_key_t k = {{tid, hi}};
    struct philox_32_4_ctr_t c = {{tid, lo, gid^tid, counter}};

    if (gid < get_num_groups(0) - 1) {
        for(int i = 0; i < repeat; ++i) {
            struct philox_32_4_ctr_t r = philox(c, k);
            generate(output, &tid, &off, &r);
        }
    } else {
        for(int i = 0; i < repeat; ++i) {
            struct philox_32_4_ctr_t r = philox(c, k);
            generate_partial(output, &tid, &numel, &off, &r);
        }
    }
}
