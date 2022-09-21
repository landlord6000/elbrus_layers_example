#include "conv_gradient_general.h"

void regroup_prm(float *__restrict__ prm, float *__restrict__ prm_new, long L, long F, long Rx, long Ry){

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
                
                    __v2di p0 = ((__v2di)((__v4sf){*(p0_p    ), *(p1_p    ), *(p2_p    ), *(p3_p    )})); // l1
                    __v2di p1 = ((__v2di)((__v4sf){*(p0_p + 1), *(p1_p + 1), *(p2_p + 1), *(p3_p + 1)})); // l2
                    __v2di p2 = ((__v2di)((__v4sf){*(p0_p + 2), *(p1_p + 2), *(p2_p + 2), *(p3_p + 2)})); // l3
                    __v2di p3 = ((__v2di)((__v4sf){*(p0_p + 3), *(p1_p + 3), *(p2_p + 3), *(p3_p + 3)})); // l4


                    *((__v2di *) (prm_new +       l*Ry*Rx*F + ry*Rx*F + rx*F + f)) = p0;
                    *((__v2di *) (prm_new + (l + 1)*Ry*Rx*F + ry*Rx*F + rx*F + f)) = p1;
                    *((__v2di *) (prm_new + (l + 2)*Ry*Rx*F + ry*Rx*F + rx*F + f)) = p2;
                    *((__v2di *) (prm_new + (l + 3)*Ry*Rx*F + ry*Rx*F + rx*F + f)) = p3;

                }

}

void regroup_map(float *__restrict__ map, float *__restrict__ map_new, long B, long Y, long X, long L){
    // B Y X L
    for(long b = 0; b < B; ++b)
        for(long y = 0; y < Y; ++y)
            for(long x = 0; x < X; ++x)
                for(long l = 0; l < L; ++l) {
                    map_new[l*B*Y*X + b*Y*X+ y*X + x] = map_new[b*Y*X*L + y*X*L + x*L + l];
                }

}

void conv_gradient_general_core1x1(float *__restrict__ in, float *__restrict__ prm, float *__restrict__ out, int L, int R) {

    __v2di a0;
    __v2di b0, b1, b2, b3, b4, b5, b6, b7;
    __v2di c0, c1, c2, c3, c4, c5, c6, c7;
    __v2di bs0, bs1, bs2, bs3, bs4, bs5, bs6, bs7;

    float val;

    val = *(float*)(out);
    c0 = float4(val);
    // bs0+=val;
    val = *(float*)(out + 1);
    c1 = float4(val);
    // bs1+=val;
    val = *(float*)(out + 2);
    c2 = float4(val);
    // bs2+=val;
    val = *(float*)(out + 3);
    c3 = float4(val);
    // bs3+=val;
    val = *(float*)(out + 4);
    c4 = float4(val);
    // bs4+=val;
    val = *(float*)(out + 5);
    c5 = float4(val);
    // bs5+=val;
    val = *(float*)(out + 6);
    c6 = float4(val);
    // bs6+=val;
    val = *(float*)(out + 7);
    c7 = float4(val);
    // bs7+=val;


 #pragma vector aligned
 #pragma ivdep
 #pragma loop count(1000)
    for(long l = 0; l < L; l+=4) {
        a0 = addr(in + l);

        b0 = addr(prm           + l);
        b1 = addr(prm + R*R*L   + l);
        b2 = addr(prm + 2*R*R*L + l);
        b3 = addr(prm + 3*R*R*L + l);
        b4 = addr(prm + 4*R*R*L + l);
        b5 = addr(prm + 5*R*R*L + l);
        b6 = addr(prm + 6*R*R*L + l);
        b7 = addr(prm + 7*R*R*L + l);

        b0 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a0, c0), b0);
        b1 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a0, c1), b1);
        b2 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a0, c2), b2);
        b3 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a0, c3), b3);
        b4 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a0, c4), b4);
        b5 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a0, c5), b5);
        b6 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a0, c6), b6);
        b7 = __builtin_e2k_qpfadds(__builtin_e2k_qpfmuls(a0, c7), b7);

        addr(prm           + l) = b0;
        addr(prm + R*R*L   + l) = b1;
        addr(prm + 2*R*R*L + l) = b2;
        addr(prm + 3*R*R*L + l) = b3;
        addr(prm + 4*R*R*L + l) = b4;
        addr(prm + 5*R*R*L + l) = b5;
        addr(prm + 6*R*R*L + l) = b6;
        addr(prm + 7*R*R*L + l) = b7;

    }

}

void conv_gradient_general_simple(float *__restrict__ in, float *__restrict__ prm, float *__restrict__ out, float *__restrict__ bs, \
    int B, int X, int Y, int Xout, int Yout, int L, int F, int R, int S, int P) {

        // float* in  = new float[B*X*Y*L];
        // float* out = new float[B*Yout*Xout*F];

        // long b = 0, y = 0, x = 0, l = 0, f = 0, rx = 0, ry = 0, ff = 0;
        memset(prm, 0.0, F*R*R*L*sizeof(float));
        #pragma omp parallel num_threads(16)
        {
        
        #pragma omp for 
                            for (long f = 0; f < F; f+=8) {    
                    for (long b = 0; b < B; b++)
                        for (long rx = 0; rx < R; rx++)
                            for (long ry = 0; ry < R; ry++) 
                                for (long x = 0; x < Xout; x++)
                                    for (long y = 0; y < Yout; y++)
                                    
                                            if (x*S+rx-P >= 0)
                                                if (y*S+ry-P >= 0)
                                                    if (x*S+rx-P < X)
                                                        if (y*S+ry-P < Y)
                                                            conv_gradient_general_core1x1( \
                                                            in + b*Y*X*L + (y*S+ry-P)*X*L + (x*S+rx-P)*L, \
                                                            prm + f*L*R*R+ry*R*L+rx*L, \
                                                            out + b*Yout*Xout*F+y*Xout*F+x*F+f, L, R);
                            bs[f] = 0;
            for (int b = 0; b < B; b++)
                for (int y = 0; y < Yout; y++)
                    for (int x = 0; x < Xout; x++) {
                        bs[f] += out[b*Yout*Xout*F+y*Xout*F+x*F+f];
                        bs[f+1] += out[b*Yout*Xout*F+y*Xout*F+x*F+f+1];
                        bs[f+2] += out[b*Yout*Xout*F+y*Xout*F+x*F+f+2];
                        bs[f+3] += out[b*Yout*Xout*F+y*Xout*F+x*F+f+3];
                        bs[f+4] += out[b*Yout*Xout*F+y*Xout*F+x*F+f+4];
                        bs[f+5] += out[b*Yout*Xout*F+y*Xout*F+x*F+f+5];
                        bs[f+6] += out[b*Yout*Xout*F+y*Xout*F+x*F+f+6];
                        bs[f+7] += out[b*Yout*Xout*F+y*Xout*F+x*F+f+7];
                    }
                            }

        }
      

    }

void conv_gradient_general(float *__restrict__ in, float *__restrict__ prm, float *__restrict__ out, float *__restrict__ bs, \
    int B, int X, int Y, int Xout, int Yout, int L, int F, int R, int S, int P) {

        // самая примитивная реализация
        conv_gradient_general_simple(in, prm, out, bs, B, X, Y, Xout, Yout, L, F, R, S, P);
       
    
    }
			
