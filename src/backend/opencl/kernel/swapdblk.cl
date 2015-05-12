/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
*/

__kernel void
swapdblk(int nb,
         __global T *dA, unsigned long dA_offset, int ldda, int inca,
         __global T *dB, unsigned long dB_offset, int lddb, int incb)
{
    const int tx = get_local_id(0);
    const int bx = get_group_id(0);

    dA += tx + bx * nb * (ldda + inca) + dA_offset;
    dB += tx + bx * nb * (lddb + incb) + dB_offset;

    T tmp;

    #pragma unroll
    for( int i = 0; i < nb; i++ ){
        tmp        = dA[i*ldda];
        dA[i*ldda] = dB[i*lddb];
        dB[i*lddb] = tmp;
    }
}
