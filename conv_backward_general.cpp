#include "conv_backward_general.h"

void regroup_prm(float *__restrict__ prm, float *__restrict__ prm_new, long L, long F, long Rx, long Ry){
 // улучшение с использованием интринсиков, не осоо быстрее стало, но небольшой прирост ест

    // for(long f = 0; f < F; ++f)
    //     for(long ry = 0; ry < Ry; ++ry)
    //         for(long rx = 0; rx < Rx; ++rx)
    //             for(long l = 0; l < L; ++l)
	// 	            prm_new[l*Ry*Rx*F + ry*Rx*F + rx*F + f]=prm[f*Ry*Rx*L + ry*Rx*L + rx*L + l];

    __v2di p0, p1, p2, p3; //p4, p5, p6, p7;
    float *p0_p, *p1_p, *p2_p, *p3_p; //*p4_p, *p5_p, *p6_p, *p7_p;

    for(long f = 0; f < F; f+=4)
        for(long ry = 0; ry < Ry; ++ry)
            for(long rx = 0; rx < Rx; ++rx)
                for(long l = 0; l < L; l+=4) {

                    p0_p = (prm       + f*Ry*Rx*L + ry*Rx*L + rx*L + l); // f1
                    p1_p = (prm + (f + 1)*Ry*Rx*L + ry*Rx*L + rx*L + l); // f2
                        p2_p = (prm + (f + 2)*Ry*Rx*L + ry*Rx*L + rx*L + l); // f3
                    p3_p = (prm + (f + 3)*Ry*Rx*L + ry*Rx*L + rx*L + l); // f4

                    // p4_p = (prm + (f + 4)*Ry*Rx*L + ry*Rx*L + rx*L + l); // f1
                    // p5_p = (prm + (f + 5)*Ry*Rx*L + ry*Rx*L + rx*L + l); // f2
                    // p6_p = (prm + (f + 6)*Ry*Rx*L + ry*Rx*L + rx*L + l); // f3
                    // p7_p = (prm + (f + 7)*Ry*Rx*L + ry*Rx*L + rx*L + l); // f4
                
                    __v2di p0 = ((__v2di)((__v4sf){*(p0_p    ), *(p1_p    ), *(p2_p    ), *(p3_p    )})); // l1
                    __v2di p1 = ((__v2di)((__v4sf){*(p0_p + 1), *(p1_p + 1), *(p2_p + 1), *(p3_p + 1)})); // l2
                    __v2di p2 = ((__v2di)((__v4sf){*(p0_p + 2), *(p1_p + 2), *(p2_p + 2), *(p3_p + 2)})); // l3
                    __v2di p3 = ((__v2di)((__v4sf){*(p0_p + 3), *(p1_p + 3), *(p2_p + 3), *(p3_p + 3)})); // l4

                    // __v2di p4 = ((__v2di)((__v4sf){*(p4_p    ), *(p5_p    ), *(p6_p    ), *(p7_p    )})); // l1
                    // __v2di p5 = ((__v2di)((__v4sf){*(p4_p + 1), *(p5_p + 1), *(p6_p + 1), *(p7_p + 1)})); // l2
                    // __v2di p6 = ((__v2di)((__v4sf){*(p4_p + 2), *(p5_p + 2), *(p6_p + 2), *(p7_p + 2)})); // l3
                    // __v2di p7 = ((__v2di)((__v4sf){*(p4_p + 3), *(p5_p + 3), *(p6_p + 3), *(p7_p + 3)})); // l4

                    *((__v2di *) (prm_new +       l*Ry*Rx*F + ry*Rx*F + rx*F + f)) = p0;
                    *((__v2di *) (prm_new + (l + 1)*Ry*Rx*F + ry*Rx*F + rx*F + f)) = p1;
                    *((__v2di *) (prm_new + (l + 2)*Ry*Rx*F + ry*Rx*F + rx*F + f)) = p2;
                    *((__v2di *) (prm_new + (l + 3)*Ry*Rx*F + ry*Rx*F + rx*F + f)) = p3;

                    // *((__v2di *) (prm_new + (l + 4)*Ry*Rx*F + ry*Rx*F + rx*F + f)) = p4;
                    // *((__v2di *) (prm_new + (l + 5)*Ry*Rx*F + ry*Rx*F + rx*F + f)) = p5;
                    // *((__v2di *) (prm_new + (l + 6)*Ry*Rx*F + ry*Rx*F + rx*F + f)) = p6;
                    // *((__v2di *) (prm_new + (l + 7)*Ry*Rx*F + ry*Rx*F + rx*F + f)) = p7;

                }

}

void core_backward_general_extended(float *__restrict__ in, float *__restrict__ prm, float *__restrict__ out,
    long F, long Rx, long Ry, long disp_out_x, long disp_out_y, long disp_in_x, long disp_in_y) {
// this core perform 4 sets of convolutions for 4 independent pixels of maps, so it is 4*16 = 64 FMA (comparing to 16 in simple core)

    __v2di a00, a01,
           a10, a11,
           a20, a21,
           a30, a31;

    __v2di b00, b01, b02, b03, b04, b05, b06, b07,
           b10, b11, b12, b13, b14, b15, b16, b17;

    __v2di c00, c01, c02, c03, c04, c05, c06, c07,
           c10, c11, c12, c13, c14, c15, c16, c17,
           c20, c21, c22, c23, c24, c25, c26, c27,
           c30, c31, c32, c33, c34, c35, c36, c37,
           c40, c41, c42, c43, c44, c45, c46, c47,
           c50, c51, c52, c53, c54, c55, c56, c57,
           c60, c61, c62, c63, c64, c65, c66, c67,
           c70, c71, c72, c73, c74, c75, c76, c77;

    __v2di temp00, temp01,
           temp10, temp11,
           temp20, temp21,
           temp30, temp31;

    temp00 = *((__v2di*) (in));
    temp01 = *((__v2di*) (in + 4));
    temp10 = *((__v2di*) (in + disp_in_x));
    temp11 = *((__v2di*) (in + disp_in_x + 4));
    temp20 = *((__v2di*) (in + disp_in_y));
    temp21 = *((__v2di*) (in + disp_in_y+ 4));
    temp30 = *((__v2di*) (in + disp_in_x + disp_in_y));
    temp31 = *((__v2di*) (in + disp_in_x + disp_in_y + 4));

    c00 = (__v2di)((__v4sf){0., 0., 0., 0.}); c10 = c00; c20 = c00; c30 = c00; c40 = c00; c50 = c00; c60 = c00; c70 = c00;
    c01 = c00;                                c11 = c00; c21 = c00; c31 = c00; c41 = c00; c51 = c00; c61 = c00; c71 = c00;
    c02 = c00;                                c12 = c00; c22 = c00; c32 = c00; c42 = c00; c52 = c00; c62 = c00; c72 = c00;
    c03 = c00;                                c13 = c00; c23 = c00; c33 = c00; c43 = c00; c53 = c00; c63 = c00; c73 = c00;
    c04 = c00;                                c14 = c00; c24 = c00; c34 = c00; c44 = c00; c54 = c00; c64 = c00; c74 = c00;
    c05 = c00;                                c15 = c00; c25 = c00; c35 = c00; c45 = c00; c55 = c00; c65 = c00; c75 = c00;
    c06 = c00;                                c16 = c00; c26 = c00; c36 = c00; c46 = c00; c56 = c00; c66 = c00; c76 = c00;
    c07 = c00;                                c17 = c00; c27 = c00; c37 = c00; c47 = c00; c57 = c00; c67 = c00; c77 = c00;

    
    

    // float* c0p = (float*)&c0;

    // printf("с00 = %f", c0p[0]);

 #pragma loop count(1000)
    for(long f = 0; f < F; f+=8) {

        a00 = *((__v2di*) (out                           + f    ));
        a01 = *((__v2di*) (out                           + f + 4));
        a10 = *((__v2di*) (out + disp_out_x              + f    ));
        a11 = *((__v2di*) (out + disp_out_x              + f + 4));
        a20 = *((__v2di*) (out + disp_out_y              + f    ));
        a21 = *((__v2di*) (out + disp_out_y              + f + 4));
        a30 = *((__v2di*) (out + disp_out_x + disp_out_y + f    ));
        a31 = *((__v2di*) (out + disp_out_x + disp_out_y + f + 4));

        b00 = *((__v2di*) (prm             + f    ));
        b01 = *((__v2di*) (prm +   Ry*Rx*F + f    ));
        b02 = *((__v2di*) (prm + 2*Ry*Rx*F + f    ));
        b03 = *((__v2di*) (prm + 3*Ry*Rx*F + f    ));
        b04 = *((__v2di*) (prm + 4*Ry*Rx*F + f    ));
        b05 = *((__v2di*) (prm + 5*Ry*Rx*F + f    ));
        b06 = *((__v2di*) (prm + 6*Ry*Rx*F + f    ));
        b07 = *((__v2di*) (prm + 7*Ry*Rx*F + f    ));
        
        b10 = *((__v2di*) (prm             + f + 4));
        b11 = *((__v2di*) (prm +   Ry*Rx*F + f + 4));
        b12 = *((__v2di*) (prm + 2*Ry*Rx*F + f + 4));
        b13 = *((__v2di*) (prm + 3*Ry*Rx*F + f + 4));
        b14 = *((__v2di*) (prm + 4*Ry*Rx*F + f + 4));
        b15 = *((__v2di*) (prm + 5*Ry*Rx*F + f + 4));
        b16 = *((__v2di*) (prm + 6*Ry*Rx*F + f + 4));
        b17 = *((__v2di*) (prm + 7*Ry*Rx*F + f + 4));

 // 1st pixel, 8 f

        c00 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a00, b00), c00);
        c01 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a00, b01), c01);
        c02 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a00, b02), c02);
        c03 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a00, b03), c03);
        c04 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a00, b04), c04);
        c05 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a00, b05), c05);
        c06 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a00, b06), c06);
        c07 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a00, b07), c07);

        c10 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a01, b10), c10);
        c11 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a01, b11), c11);
        c12 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a01, b12), c12);
        c13 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a01, b13), c13);
        c14 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a01, b14), c14);
        c15 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a01, b15), c15);
        c16 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a01, b16), c16);
        c17 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a01, b17), c17);

 // 2nd pixel ( -> OX)

        c20 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a10, b00), c20);
        c21 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a10, b01), c21);
        c22 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a10, b02), c22);
        c23 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a10, b03), c23);
        c24 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a10, b04), c24);
        c25 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a10, b05), c25);
        c26 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a10, b06), c26);
        c27 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a10, b07), c27);

        c30 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a11, b10), c30);
        c31 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a11, b11), c31);
        c32 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a11, b12), c32);
        c33 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a11, b13), c33);
        c34 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a11, b14), c34);
        c35 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a11, b15), c35);
        c36 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a11, b16), c36);
        c37 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a11, b17), c37);

 // 3nd pixel ( -> OY)

        c40 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a20, b00), c40);
        c41 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a20, b01), c41);
        c42 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a20, b02), c42);
        c43 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a20, b03), c43);
        c44 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a20, b04), c44);
        c45 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a20, b05), c45);
        c46 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a20, b06), c46);
        c47 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a20, b07), c47);

        c50 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a21, b10), c50);
        c51 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a21, b11), c51);
        c52 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a21, b12), c52);
        c53 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a21, b13), c53);
        c54 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a21, b14), c54);
        c55 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a21, b15), c55);
        c56 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a21, b16), c56);
        c57 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a21, b17), c57);

 // 4nd pixel ( -> OY -> OX)

        c60 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a30, b00), c60);
        c61 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a30, b01), c61);
        c62 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a30, b02), c62);
        c63 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a30, b03), c63);
        c64 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a30, b04), c64);
        c65 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a30, b05), c65);
        c66 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a30, b06), c66);
        c67 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a30, b07), c67);

        c70 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a31, b10), c70);
        c71 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a31, b11), c71);
        c72 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a31, b12), c72);
        c73 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a31, b13), c73);
        c74 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a31, b14), c74);
        c75 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a31, b15), c75);
        c76 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a31, b16), c76);
        c77 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a31, b17), c77);

    }

    *((__v2di *)(in))     = __builtin_e2k_qpfadds(__builtin_e2k_qpfhadds(
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c00, c10), __builtin_e2k_qpfadds(c01, c11)), 
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c02, c12), __builtin_e2k_qpfadds(c03, c13))), temp00);

    *((__v2di *)(in + 4)) = __builtin_e2k_qpfadds(__builtin_e2k_qpfhadds(
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c04, c14), __builtin_e2k_qpfadds(c05, c15)), 
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c06, c16), __builtin_e2k_qpfadds(c07, c17))), temp01);


    *((__v2di *)(in + disp_in_x))     = __builtin_e2k_qpfadds(__builtin_e2k_qpfhadds(
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c20, c30), __builtin_e2k_qpfadds(c21, c31)), 
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c22, c32), __builtin_e2k_qpfadds(c23, c33))), temp10);

    *((__v2di *)(in + disp_in_x + 4)) = __builtin_e2k_qpfadds(__builtin_e2k_qpfhadds(
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c24, c34), __builtin_e2k_qpfadds(c25, c35)), 
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c26, c36), __builtin_e2k_qpfadds(c27, c37))), temp11);


    *((__v2di *)(in + disp_in_y))     = __builtin_e2k_qpfadds(__builtin_e2k_qpfhadds(
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c40, c50), __builtin_e2k_qpfadds(c41, c51)), 
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c42, c52), __builtin_e2k_qpfadds(c43, c53))), temp20);

    *((__v2di *)(in + disp_in_y + 4)) = __builtin_e2k_qpfadds(__builtin_e2k_qpfhadds(
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c44, c54), __builtin_e2k_qpfadds(c45, c55)), 
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c46, c56), __builtin_e2k_qpfadds(c47, c57))), temp21);


    *((__v2di *)(in + disp_in_x + disp_in_y))     = __builtin_e2k_qpfadds(__builtin_e2k_qpfhadds(
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c60, c70), __builtin_e2k_qpfadds(c61, c71)), 
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c62, c72), __builtin_e2k_qpfadds(c63, c73))), temp30);

    *((__v2di *)(in + disp_in_x + disp_in_y + 4)) = __builtin_e2k_qpfadds(__builtin_e2k_qpfhadds(
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c64, c74), __builtin_e2k_qpfadds(c65, c75)), 
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c66, c76), __builtin_e2k_qpfadds(c67, c77))), temp31);
}

void core_backward_general(float *__restrict__ in, float *__restrict__ prm, float *__restrict__ out, long F, long Rx, long Ry) {

    __v2di a00, a01;

    __v2di b00, b01, b02, b03, b04, b05, b06, b07,
           b10, b11, b12, b13, b14, b15, b16, b17;

    __v2di c00, c01, c02, c03, c04, c05, c06, c07,
           c10, c11, c12, c13, c14, c15, c16, c17;

    __v2di temp0, temp1;

    temp0 = *((__v2di*) (in));
    temp1 = *((__v2di*) (in + 4));

    c00 = (__v2di)((__v4sf){0., 0., 0., 0.});
    c01 = c00;
    c02 = c00;
    c03 = c00;
    c04 = c00;
    c05 = c00;
    c06 = c00;
    c07 = c00;

    c10 = c00;
    c11 = c00;
    c12 = c00;
    c13 = c00;
    c14 = c00;
    c15 = c00;
    c16 = c00;
    c17 = c00;

    // float* c0p = (float*)&c0;

    // printf("с00 = %f", c0p[0]);

 #pragma loop count(1000)
    for(long f = 0; f < F; f+=8) {
        // можно увеличить по f до +=8
        a00 = *((__v2di*) (out + f));
        a01 = *((__v2di*) (out + f + 4));

        b00 = *((__v2di*) (prm             + f    ));
        b01 = *((__v2di*) (prm +   Ry*Rx*F + f    ));
        b02 = *((__v2di*) (prm + 2*Ry*Rx*F + f    ));
        b03 = *((__v2di*) (prm + 3*Ry*Rx*F + f    ));
        b04 = *((__v2di*) (prm + 4*Ry*Rx*F + f    ));
        b05 = *((__v2di*) (prm + 5*Ry*Rx*F + f    ));
        b06 = *((__v2di*) (prm + 6*Ry*Rx*F + f    ));
        b07 = *((__v2di*) (prm + 7*Ry*Rx*F + f    ));
        
        b10 = *((__v2di*) (prm             + f + 4));
        b11 = *((__v2di*) (prm +   Ry*Rx*F + f + 4));
        b12 = *((__v2di*) (prm + 2*Ry*Rx*F + f + 4));
        b13 = *((__v2di*) (prm + 3*Ry*Rx*F + f + 4));
        b14 = *((__v2di*) (prm + 4*Ry*Rx*F + f + 4));
        b15 = *((__v2di*) (prm + 5*Ry*Rx*F + f + 4));
        b16 = *((__v2di*) (prm + 6*Ry*Rx*F + f + 4));
        b17 = *((__v2di*) (prm + 7*Ry*Rx*F + f + 4));

        // float* temp0p = (float*)&temp0;
        // float* temp1p = (float*)&temp1;
        // printf("temp0 = %f %f %f %f\n", temp0p[0], temp0p[1], temp0p[2], temp0p[3]);
        // printf("temp1 = %f %f %f %f\n", temp1p[0], temp1p[1], temp1p[2], temp1p[3]);

        // printf("b00 = %f, b01 = %f b02 = %f b03 = %f\n", *(float*)&b0, *((float*)&b0 + 1), *((float*)&b0 + 2), *((float*)&b0 + 3));
        // printf("b10 = %f, b11 = %f b12 = %f b13 = %f\n", *(float*)&b1, *((float*)&b1 + 1), *((float*)&b1 + 2), *((float*)&b1 + 3));
        // printf("b20 = %f, b11 = %f b12 = %f b13 = %f\n", *(float*)&b2, *((float*)&b2 + 1), *((float*)&b2 + 2), *((float*)&b2 + 3));
        // printf("b30 = %f, b11 = %f b12 = %f b13 = %f\n", *(float*)&b3, *((float*)&b3 + 1), *((float*)&b3 + 2), *((float*)&b3 + 3));
        // printf("b40 = %f, b11 = %f b12 = %f b13 = %f\n", *(float*)&b4, *((float*)&b4 + 1), *((float*)&b4 + 2), *((float*)&b4 + 3));
        // printf("b50 = %f, b11 = %f b12 = %f b13 = %f\n", *(float*)&b5, *((float*)&b5 + 1), *((float*)&b5 + 2), *((float*)&b5 + 3));
        // printf("b60 = %f, b11 = %f b12 = %f b13 = %f\n", *(float*)&b6, *((float*)&b6 + 1), *((float*)&b6 + 2), *((float*)&b6 + 3));
        // printf("b70 = %f, b11 = %f b12 = %f b13 = %f\n\n", *(float*)&b7, *((float*)&b7 + 1), *((float*)&b7 + 2), *((float*)&b7 + 3));

        c00 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a00, b00), c00);
        c01 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a00, b01), c01);
        c02 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a00, b02), c02);
        c03 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a00, b03), c03);
        c04 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a00, b04), c04);
        c05 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a00, b05), c05);
        c06 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a00, b06), c06);
        c07 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a00, b07), c07);

        c10 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a01, b10), c10);
        c11 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a01, b11), c11);
        c12 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a01, b12), c12);
        c13 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a01, b13), c13);
        c14 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a01, b14), c14);
        c15 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a01, b15), c15);
        c16 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a01, b16), c16);
        c17 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a01, b17), c17);

        // printf("c00 = %f, b01 = %f b02 = %f b03 = %f\n", *(float*)&c0, *((float*)&c0 + 1), *((float*)&c0 + 2), *((float*)&c0 + 3));
        // printf("c10 = %f, b11 = %f b12 = %f b13 = %f\n", *(float*)&c1, *((float*)&c1 + 1), *((float*)&c1 + 2), *((float*)&c1 + 3));
        // printf("c20 = %f, b11 = %f b12 = %f b13 = %f\n", *(float*)&c2, *((float*)&c2 + 1), *((float*)&c2 + 2), *((float*)&c2 + 3));
        // printf("c30 = %f, b11 = %f b12 = %f b13 = %f\n", *(float*)&c3, *((float*)&c3 + 1), *((float*)&c3 + 2), *((float*)&c3 + 3));
        // printf("c40 = %f, b11 = %f b12 = %f b13 = %f\n", *(float*)&c4, *((float*)&c4 + 1), *((float*)&c4 + 2), *((float*)&c4 + 3));
        // printf("c50 = %f, b11 = %f b12 = %f b13 = %f\n", *(float*)&c5, *((float*)&c5 + 1), *((float*)&c5 + 2), *((float*)&c5 + 3));
        // printf("c60 = %f, b11 = %f b12 = %f b13 = %f\n", *(float*)&c6, *((float*)&c6 + 1), *((float*)&c6 + 2), *((float*)&c6 + 3));
        // printf("c70 = %f, b11 = %f b12 = %f b13 = %f\n", *(float*)&c7, *((float*)&c7 + 1), *((float*)&c7 + 2), *((float*)&c7 + 3));

    }

    *((__v2di *)(in))     = __builtin_e2k_qpfadds(__builtin_e2k_qpfhadds(
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c00, c10), __builtin_e2k_qpfadds(c01, c11)), 
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c02, c12), __builtin_e2k_qpfadds(c03, c13))), temp0);
    *((__v2di *)(in + 4)) = __builtin_e2k_qpfadds(__builtin_e2k_qpfhadds(
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c04, c14), __builtin_e2k_qpfadds(c05, c15)), 
        __builtin_e2k_qpfhadds(
        __builtin_e2k_qpfadds(c06, c16), __builtin_e2k_qpfadds(c07, c17))), temp1);
}


void conv_backward_general(float *__restrict__ in, float *__restrict__ prm, float *__restrict__ prm_new, float *__restrict__ out, \
    int B, int X, int Y, int Xout, int Yout, int L, int F, int Rx, int Ry, int S, int Px, int Py) {

    regroup_prm(prm, prm_new, L, F, Rx, Ry); // put prm F*Ry*Rx*L ->  L*Ry*Rx*F
    memset(in, 0., B*Y*X*L*sizeof(float));  // in = 0

    
    long disp_x = ceil((double)Rx / (double)S); // количество шагов S, которые мы делаем в одном блоке, чтобы дойти до 2, 3 и 4 пикселя
    long disp_y = ceil((double)Ry / (double)S); 

    long Xcore_count = 0, Ycore_count = 0, Xcore = 0, Ycore = 0, \
         X_pad = 0, Y_pad = 0, X_pad_shift = 0, Y_pad_shift = 0;
    bool fl = false;
    
    if(S == 1) {
        
        if(Px == 0) {
            Xcore_count = (X - (3*Rx - 1)) / (2*Rx) + 1;
            Ycore_count = (Y - (3*Ry - 1)) / (2*Ry) + 1;
            if((X >= 3*Rx - 1) && (Y >= 3*Ry - 1)) {
                fl = true;
            }
        }
        else {
            X_pad_shift = 0;
            Y_pad_shift = 0;
            X_pad = X - 2*X_pad_shift;
            Y_pad = Y - 2*Y_pad_shift;
            Xcore_count = (X_pad - (3*Rx - 1)) / (2*Rx) + 1;
            Ycore_count = (Y_pad - (3*Ry - 1)) / (2*Ry) + 1;
            if((X_pad >= 3*Rx - 1) && (Y_pad >= 3*Ry - 1)) {
                fl = true;
            }
        }
    }
    else if(S == 2) {
        
        if(Px == 0) {
            Xcore_count = (X - (3*Rx + 2*(Rx % 2) - 2)) / (2*Rx + 2*(Rx % 2)) + 1;
            Ycore_count = (Y - (3*Ry + 2*(Ry % 2) - 2)) / (2*Ry + 2*(Ry % 2)) + 1;
            if((X >= 3*Rx + 2*(Rx % 2) - 2) && (Y >= 3*Ry + 2*(Ry % 2) - 2)) {
                fl = true;
            }
        }
        else {
            X_pad_shift = Px;
            Y_pad_shift = Py;
            X_pad = X - 2*X_pad_shift;
            Y_pad = Y - 2*Y_pad_shift;
            Xcore_count = (X_pad - (3*Rx + 2*(Rx % 2) - 2)) / (2*Rx + 2*(Rx % 2)) + 1;
            Ycore_count = (Y_pad - (3*Ry + 2*(Ry % 2) - 2)) / (2*Ry + 2*(Ry % 2)) + 1;
            if((X_pad >= 3*Rx + 2*(Rx % 2) - 2) && (Y_pad >= 3*Ry + 2*(Ry % 2) - 2)) {
                fl = true;
            }
        }
    }
    else if(S == 3) {

        if(Px == 0) {
            Xcore_count = (X - (2*Rx + ((3 - Rx % 3) % 3) + ((Rx - 1) / 3) * 3)) / (ceil(Rx / 3.)*6) + 1;
            Ycore_count = (Y - (2*Ry + ((3 - Ry % 3) % 3) + ((Ry - 1) / 3) * 3)) / (ceil(Ry / 3.)*6) + 1;;
            if((X >= 2*Rx + ((3 - Rx % 3) % 3) + ((Rx - 1) / 3) * 3) && (Y >= 2*Ry + ((3 - Ry % 3) % 3) + ((Ry - 1) / 3) * 3)) {
                fl = true;
            }
        }
        else {
            X_pad_shift = 2*Px;
            Y_pad_shift = 2*Py;
            X_pad = X - 2*X_pad_shift;
            Y_pad = Y - 2*Y_pad_shift;
            Xcore_count = (X_pad - (2*Rx + ((3 - Rx % 3) % 3) + ((Rx - 1) / 3) * 3)) / (ceil(Rx / 3.)*6) + 1;
            Ycore_count = (Y_pad - (2*Ry + ((3 - Ry % 3) % 3) + ((Ry - 1) / 3) * 3)) / (ceil(Ry / 3.)*6) + 1;;
            if((X_pad >= 2*Rx + ((3 - Rx % 3) % 3) + ((Rx - 1) / 3) * 3) && (Y_pad >= 2*Ry + ((3 - Ry % 3) % 3) + ((Ry - 1) / 3) * 3)) {
                fl = true;
            }
        }
    }

    int np = 16;
    long L2_size = 512 * 1024 / 4; // 512 KB
    long L_block = (L2_size - 8*F*S*S)*0.5 / (F + 4*Rx*Ry + 2*Rx*S + S*S);
    // printf("L_block = %d\n", L_block);
    for(long l = 0; l <= L + 8; l+=8) {
        if((l - L_block) > 0) {
            L_block = l - 8;
            break;
        }
    }
    long L_bigg = (L / L_block) * L_block;
    // long L_rest = L - L_bigg;
    // printf("L = %d L_bigg = %d L_block = %d L_rest = %d\n", L, L_bigg, L_block, L_rest);

    #pragma omp parallel num_threads(np) 
    {
        if(fl) {
            if(Px != 0) {
                // pad via x_upper
            
                #pragma omp for
                for(int b = 0; b < B; b++)
                    for(int l = 0; l < L; l+=8)
                        for(int x = 0; x < Xout; x++)
                            for(int y = 0; y < Py; y++)
                                for(int rx = 0; rx < Rx; rx++){
                                    if (x*S+rx-Px >= 0)
                                    if (x*S+rx-Px < X)
                                        for(int ry = 0; ry  < Ry; ry++) 
                                            if (y*S+ry-Py >= 0)
                                            if (y*S+ry-Py < Y) {
                                                core_backward_general(in + b*Y*X*L + (y*S + ry - Py)*X*L + (x*S + rx - Px)*L + l, prm_new + l*Ry*Rx*F + ry*Rx*F + rx*F, \
                                                                        out + b*Yout*Xout*F + y*Xout*F + x*F, F, Rx, Ry);   
                                            }
                                }

                // pad via x_down
                #pragma omp for
                for(int b = 0; b < B; b++)
                    for(int l = 0; l < L; l+=8)
                        for(int x = 0; x < Xout; x++)
                            for(int y = Yout - Py; y < Yout; y++)
                                for(int rx = 0; rx < Rx; rx++) {
                                    if (x*S+rx-Px >= 0)
                                    if (x*S+rx-Px < X)
                                        for(int ry = 0; ry  < Ry; ry++)
                                            if (y*S+ry-Py >= 0)
                                            if (y*S+ry-Py < Y) {
                                                core_backward_general(in + b*Y*X*L + (y*S + ry - Py)*X*L + (x*S + rx - Px)*L + l, prm_new + l*Ry*Rx*F + ry*Rx*F + rx*F, \
                                                                        out + b*Yout*Xout*F + y*Xout*F + x*F, F, Rx, Ry);   
                                            }
                                }

                // pad via y_left
                #pragma omp for schedule (auto)
                for(int b = 0; b < B; b++)
                    for(int l = 0; l < L; l+=8)
                        for(int x = 0; x < Px; x++)
                            for(int y = Py; y < Yout - Py; y++)
                                for(int rx = 0; rx < Rx; rx++) {
                                    if (x*S+rx-Px >= 0)
                                    if (x*S+rx-Px < X)
                                        for(int ry = 0; ry  < Ry; ry++)
                                            if (y*S+ry-Py >= 0)
                                            if (y*S+ry-Py < Y) {
                                                core_backward_general(in + b*Y*X*L + (y*S + ry - Py)*X*L + (x*S + rx - Px)*L + l, prm_new + l*Ry*Rx*F + ry*Rx*F + rx*F, \
                                                                        out + b*Yout*Xout*F + y*Xout*F + x*F, F, Rx, Ry);   
                                            }
                                }
                
                // pad via y_right
                #pragma omp for
                for(int b = 0; b < B; b++)
                    for(int l = 0; l < L; l+=8)
                        for(int x = Xout - Px; x < Xout; x++)
                            for(int y = Py; y < Yout - Py; y++)
                                for(int rx = 0; rx < Rx; rx++) {
                                    if (x*S+rx-Px >= 0)
                                    if (x*S+rx-Px < X)
                                        for(int ry = 0; ry  < Ry; ry++)
                                            if (y*S+ry-Py >= 0)
                                            if (y*S+ry-Py < Y) {
                                                core_backward_general(in + b*Y*X*L + (y*S + ry - Py)*X*L + (x*S + rx - Px)*L + l, prm_new + l*Ry*Rx*F + ry*Rx*F + rx*F, \
                                                                        out + b*Yout*Xout*F + y*Xout*F + x*F, F, Rx, Ry);   
                                            }
                                }
            }


            // shifting all float* out  from padding
            float *in_s  = in  + Y_pad_shift*X*L + X_pad_shift*L;
            float *out_s = out + Py*Xout*F + Px*F;


            // main core
            // #pragma omp for //schedule (auto)
            // for(long b = 0; b < B; ++b)
            //     for(long l = 0; l < L; l+=8)
            //         for(long x = 0; x < Xcore_count; x++) 
            //             for(long y = 0; y < Ycore_count; y++)
            //                 for(long ry = 0; ry < Ry; ++ry)
            //                     for(long rx = 0; rx < Rx; ++rx) {
            //                         for(long i = 0; i < disp_y; ++i) 
            //                             for(long j = 0; j < disp_x; ++j) {
            //                                 core_backward_general_extended(\
            //                                 in_s + b*Y*X*L + (y*disp_y*S*2 + ry + i*S)*X*L + (x*disp_x*S*2 + rx + j*S)*L + l,\
            //                                 prm_new + l*Ry*Rx*F + ry*Rx*F + rx*F, \
            //                                 out_s + b*Yout*Xout*F + (y*disp_y*2 + i)*Xout*F + (x*disp_x*2 + j)*F, F, Rx, Ry, disp_x*F, disp_y*Xout*F, disp_x*L*S, disp_y*X*L*S);
            //                         }
            //                     }

         
            // L_block = 312;
            // L_bigg = (L / L_block) * L_block;
          
            // main core   
            #pragma omp for schedule (auto)
            for(long b = 0; b < B; ++b)
                for(long l = 0; l < L_bigg; l+=L_block)
                    for(long x = 0; x < Xcore_count; x++) 
                        for(long y = 0; y < Ycore_count; y++)
                            for(long ll = 0; ll < L_block; ll+=8)
                                for(long ry = 0; ry < Ry; ++ry)
                                    for(long rx = 0; rx < Rx; ++rx) {
                                        for(long i = 0; i < disp_y; ++i) 
                                            for(long j = 0; j < disp_x; ++j) {
                                                // printf("l = %d  ll = %d\n", l, ll);
                                                core_backward_general_extended(\
                                                in_s + b*Y*X*L + (y*disp_y*S*2 + ry + i*S)*X*L + (x*disp_x*S*2 + rx + j*S)*L + (l + ll),\
                                                prm_new + (l + ll)*Ry*Rx*F + ry*Rx*F + rx*F, \
                                                out_s + b*Yout*Xout*F + (y*disp_y*2 + i)*Xout*F + (x*disp_x*2 + j)*F, F, Rx, Ry, disp_x*F, disp_y*Xout*F, disp_x*L*S, disp_y*X*L*S);
                                        }
                                    }
            if(L - L_bigg) {
                #pragma omp for schedule (auto)
                for(long b = 0; b < B; ++b)
                    for(long x = 0; x < Xcore_count; x++) 
                        for(long y = 0; y < Ycore_count; y++)
                            for(long l = L_bigg; l < L; l+=8)
                                for(long ry = 0; ry < Ry; ++ry)
                                    for(long rx = 0; rx < Rx; ++rx) {
                                        for(long i = 0; i < disp_y; ++i) 
                                            for(long j = 0; j < disp_x; ++j) {
                                                // printf("l = %d  ll = %d\n", l, ll);
                                                core_backward_general_extended(\
                                                in_s + b*Y*X*L + (y*disp_y*S*2 + ry + i*S)*X*L + (x*disp_x*S*2 + rx + j*S)*L + l,\
                                                prm_new + l*Ry*Rx*F + ry*Rx*F + rx*F, \
                                                out_s + b*Yout*Xout*F + (y*disp_y*2 + i)*Xout*F + (x*disp_x*2 + j)*F, F, Rx, Ry, disp_x*F, disp_y*Xout*F, disp_x*L*S, disp_y*X*L*S);
                                        }
                                    }
            }

            // long X_block = 2;
            // long X_bigg = (Xcore_count / X_block) * X_block;
            // long Y_block = 2;
            // long Y_bigg = (Ycore_count / Y_block) * Y_block;
            // printf("Xcore_count = %d Ycore_count = %d\n", Xcore_count, Ycore_count);
            // printf("X_block = %d Y_block = %d\n", X_block, Y_block);
            // printf("X_bigg = %d, Y_bigg = %d\n", X_bigg, Y_bigg);
          
            // // main core   
            // #pragma omp for schedule (auto)
            // for(long b = 0; b < B; ++b)
            //     for(long x = 0; x < X_bigg; x+=X_block) 
            //         for(long y = 0; y < Y_bigg; y+=Y_block)
            //             for(long l = 0; l < L; l++)
            //                 for(long xx = 0; xx < X_block; xx++) 
            //                     for(long yy = 0; yy < Y_block; yy++)
            //                         for(long ry = 0; ry < Ry; ++ry)
            //                             for(long rx = 0; rx < Rx; ++rx) {
            //                                 for(long i = 0; i < disp_y; ++i) 
            //                                     for(long j = 0; j < disp_x; ++j) {
            //                                         core_backward_general_extended(\
            //                                         in_s + b*Y*X*L + ((y + yy)*disp_y*S*2 + ry + i*S)*X*L + ((x + xx)*disp_x*S*2 + rx + j*S)*L + l,\
            //                                         prm_new + l*Ry*Rx*F + ry*Rx*F + rx*F, \
            //                                         out_s + b*Yout*Xout*F + ((y + yy)*disp_y*2 + i)*Xout*F + ((x + xx)*disp_x*2 + j)*F, F, Rx, Ry, disp_x*F, disp_y*Xout*F, disp_x*L*S, disp_y*X*L*S);
            //                                 }
            //                             }
            // if(Xcore_count - X_bigg) {
            //     #pragma omp for schedule (auto)
            //     for(long b = 0; b < B; ++b)
            //         for(long l = 0; l < L; l+=8)
            //             for(long x = X_bigg; x < Xcore_count; x++) 
            //                 for(long y = 0; y < Y_bigg; y++)
            //                     for(long ry = 0; ry < Ry; ++ry)
            //                         for(long rx = 0; rx < Rx; ++rx) {
            //                             for(long i = 0; i < disp_y; ++i) 
            //                                 for(long j = 0; j < disp_x; ++j) {
            //                                     // printf("l = %d  ll = %d\n", l, ll);
            //                                     core_backward_general_extended(\
            //                                     in_s + b*Y*X*L + (y*disp_y*S*2 + ry + i*S)*X*L + (x*disp_x*S*2 + rx + j*S)*L + l,\
            //                                     prm_new + l*Ry*Rx*F + ry*Rx*F + rx*F, \
            //                                     out_s + b*Yout*Xout*F + (y*disp_y*2 + i)*Xout*F + (x*disp_x*2 + j)*F, F, Rx, Ry, disp_x*F, disp_y*Xout*F, disp_x*L*S, disp_y*X*L*S);
            //                             }
            //                         }
            // }

            // if(Xcore_count - X_bigg) {
            //     #pragma omp for schedule (auto)
            //     for(long b = 0; b < B; ++b)
            //         for(long l = 0; l < L; l+=8)
            //             for(long y = Y_bigg; y < Ycore_count; y++) 
            //                 for(long x = 0; x < X_bigg; x++)
            //                     for(long ry = 0; ry < Ry; ++ry)
            //                         for(long rx = 0; rx < Rx; ++rx) {
            //                             for(long i = 0; i < disp_y; ++i) 
            //                                 for(long j = 0; j < disp_x; ++j) {
            //                                     // printf("l = %d  ll = %d\n", l, ll);
            //                                     core_backward_general_extended(\
            //                                     in_s + b*Y*X*L + (y*disp_y*S*2 + ry + i*S)*X*L + (x*disp_x*S*2 + rx + j*S)*L + l,\
            //                                     prm_new + l*Ry*Rx*F + ry*Rx*F + rx*F, \
            //                                     out_s + b*Yout*Xout*F + (y*disp_y*2 + i)*Xout*F + (x*disp_x*2 + j)*F, F, Rx, Ry, disp_x*F, disp_y*Xout*F, disp_x*L*S, disp_y*X*L*S);
            //                             }
            //                         }
            // }

            // if(Xcore_count - X_bigg && Ycore_count - Y_bigg) {
            //     #pragma omp for schedule (auto)
            //     for(long b = 0; b < B; ++b)
            //         for(long l = 0; l < L; l+=8)
            //             for(long y = Y_bigg; y < Ycore_count; y++) 
            //                 for(long x = X_bigg; x < Xcore_count; x++)
            //                     for(long ry = 0; ry < Ry; ++ry)
            //                         for(long rx = 0; rx < Rx; ++rx) {
            //                             for(long i = 0; i < disp_y; ++i) 
            //                                 for(long j = 0; j < disp_x; ++j) {
            //                                     // printf("l = %d  ll = %d\n", l, ll);
            //                                     core_backward_general_extended(\
            //                                     in_s + b*Y*X*L + (y*disp_y*S*2 + ry + i*S)*X*L + (x*disp_x*S*2 + rx + j*S)*L + l,\
            //                                     prm_new + l*Ry*Rx*F + ry*Rx*F + rx*F, \
            //                                     out_s + b*Yout*Xout*F + (y*disp_y*2 + i)*Xout*F + (x*disp_x*2 + j)*F, F, Rx, Ry, disp_x*F, disp_y*Xout*F, disp_x*L*S, disp_y*X*L*S);
            //                             }
            //                         }
            // }
        
        

            // tail via Y
            #pragma omp for schedule (auto)
            for(long b = 0; b < B; ++b)
                for(long l = 0; l < L; l+=8)
                    for(long x = Xcore_count*disp_x*2; x < Xout - 2*Px; ++x) 
                        for(long y = 0; y < Yout - 2*Py; ++y)
                            for(long ry = 0; ry < Ry; ++ry)
                                for(long rx = 0; rx < Rx; ++rx){
                                        core_backward_general(\
                                        in_s + b*Y*X*L + (y*S + ry)*X*L + (x*S + rx)*L + l,\
                                        prm_new + l*Ry*Rx*F + ry*Rx*F + rx*F, \
                                        out_s + b*Yout*Xout*F + y*Xout*F + x*F, F, Rx, Ry);
                                }
            // tail via X    
            #pragma omp for schedule (auto)
            for(long b = 0; b < B; ++b)
                for(long l = 0; l < L; l+=8)
                    for(long x = 0; x < Xcore_count*disp_x*2; ++x)
                        for(long y = Ycore_count*disp_y*2; y < Yout - 2*Py; ++y)
                            for(long ry = 0; ry < Ry; ++ry)
                                for(long rx = 0; rx < Rx; ++rx){
                                        core_backward_general(\
                                        in_s + b*Y*X*L + (y*S + ry)*X*L + (x*S + rx)*L + l,\
                                        prm_new + l*Ry*Rx*F + ry*Rx*F + rx*F, \
                                        out_s + b*Yout*Xout*F + y*Xout*F + x*F, F, Rx, Ry);
                                }
        }
        else {
            // printf("here!!!\n");
            // // main core  
            // #pragma omp for schedule (auto)
            // for(int b = 0; b < B; b++)
            //     for(int l = 0; l < L; l+=8) {
            //         for(int x = 0; x < Xout; x++)
            //             for(int y = 0; y < Yout; y++)
            //                 for(int rx = 0; rx < Rx; rx++)
            //                     for(int ry = 0; ry  < Ry; ry++)
            //                         if (x*S+rx-Px >= 0)
            //                         if (y*S+ry-Py >= 0)
            //                         if (x*S+rx-Px < X)
            //                         if (y*S+ry-Py < Y) {
            //                             core_backward_general(in + b*Y*X*L + (y*S + ry - Py)*X*L + (x*S + rx - Px)*L + l, prm_new + l*Ry*Rx*F + ry*Rx*F + rx*F, \
            //                                                     out + b*Yout*Xout*F + y*Xout*F + x*F, F, Rx, Ry);   
            //                         }
            //     }

            // #pragma omp for 
            // for(int b = 0; b < B; b++)
            //     for(int x = 0; x < Xout; x++)
            //         for(int y = 0; y < Yout; y++)
            //             for(int l = 0; l < L; l+=8) {
            //                 for(int rx = 0; rx < Rx; rx++)
            //                     for(int ry = 0; ry  < Ry; ry++)
            //                         if (x*S+rx-Px >= 0)
            //                         if (y*S+ry-Py >= 0)
            //                         if (x*S+rx-Px < X)
            //                         if (y*S+ry-Py < Y) {
            //                             core_backward_general(in + b*Y*X*L + (y*S + ry - Py)*X*L + (x*S + rx - Px)*L + l, prm_new + l*Ry*Rx*F + ry*Rx*F + rx*F, \
            //                                                     out + b*Yout*Xout*F + y*Xout*F + x*F, F, Rx, Ry);   
            //                         }
            //     }


        //     L_block = 16;
        //     // for(long l = 0; l <= L + 8; l+=8) {
        //     //     if((l - L_block) > 0) {
        //     //         L_block = l - 8;
        //     //         break;
        //     //     }
        //     // }
        //     long L_bigg = (L / L_block) * L_block;

        //   main core  
            #pragma omp for schedule (auto)
            for(long b = 0; b < B; ++b)
                for(long l = 0; l < L_bigg; l+=L_block)
                    for(long x = 0; x < Xout; x++) 
                        for(long y = 0; y < Yout; y++)
                            for(long ll = 0; ll < L_block; ll+=8)
                                for(int rx = 0; rx < Rx; rx++)
                                    for(int ry = 0; ry  < Ry; ry++)
                                        if (x*S+rx-Px >= 0)
                                        if (y*S+ry-Py >= 0)
                                        if (x*S+rx-Px < X)
                                        if (y*S+ry-Py < Y) {
                                            core_backward_general(in + b*Y*X*L + (y*S + ry - Py)*X*L + (x*S + rx - Px)*L + (l + ll), prm_new + (l + ll)*Ry*Rx*F + ry*Rx*F + rx*F, \
                                                                    out + b*Yout*Xout*F + y*Xout*F + x*F, F, Rx, Ry);   
                                        }

            if(L - L_bigg) {
                #pragma omp for schedule (auto)
                for(long b = 0; b < B; ++b)
                    for(long y = 0; y < Yout; y++)
                        for(long l = L_bigg; l < L; l+=8)
                            for(long x = 0; x < Xout; x++) 
                                for(int rx = 0; rx < Rx; rx++)
                                    for(int ry = 0; ry  < Ry; ry++)
                                        if (x*S+rx-Px >= 0)
                                        if (y*S+ry-Py >= 0)
                                        if (x*S+rx-Px < X)
                                        if (y*S+ry-Py < Y) {
                                            core_backward_general(in + b*Y*X*L + (y*S + ry - Py)*X*L + (x*S + rx - Px)*L + l, prm_new + l*Ry*Rx*F + ry*Rx*F + rx*F, \
                                                                    out + b*Yout*Xout*F + y*Xout*F + x*F, F, Rx, Ry);   
                                        }    
            }
        }
    }
    }
    
    // core perf test (we want to define peak perf on r1s1p0)
    // core_backward_general_extended(in, prm_new, out, F, Rx, Ry, F, Xout*F, L, X*L);
			
