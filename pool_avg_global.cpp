#include "pool_avg_global.h"

void pool_avg_global_core11_init(float *__restrict__ in, float *__restrict__ out, long L) {

	__v2di a00, out_in00, out_out00;

	for(long l = 0; l < L; l+=4) {

		a00 = *((__v2di*)(in  + l));

		*((__v2di*)(out + l)) = a00;
	}

}

void pool_avg_global_core11(float *__restrict__ in, float *__restrict__ out, long L) {

	__v2di a00, out_in00, out_out00;

	for(long l = 0; l < L; l+=4) {

		a00      = *((__v2di*)(in  + l));
		out_in00 = *((__v2di*)(out + l));

		out_out00 = __builtin_e2k_qpfadds(a00, out_in00);
		*((__v2di*)(out + l)) = out_out00;
	}

}

void pool_avg_global_core11_last(float *__restrict__ in, float *__restrict__ out, float* XY, long L) {

	__v2di a00, out_in00, out_out00;
	__v2di dXY0 = (__v2di)(__builtin_ia32_vbroadcastss(XY));
	// dXY0 = (__v2di)__builtin_ia32_rcpps((__v4sf)(dXY0));

	for(long l = 0; l < L; l+=4) {

		a00      = *((__v2di*)(in     + l));
		out_in00 = *((__v2di*)(out + l));

		out_out00 = __builtin_e2k_qpfadds(a00, out_in00);
		*((__v2di*)(out + l)) = __builtin_e2k_qpfmuls(out_out00, dXY0);
	}

}

void pool_avg_global_core33(float *__restrict__ in, float *__restrict__ out, float* XY, long X, long L) {

	__v2di a00, a01, a02,
		   a10, a11, a12,
		   a20, a21, a22;

	__v2di b0, b1, b2, b3, b4, b5, b6, b7;

	__v2di c0, c1;

	__v2di dXY0 = (__v2di)(__builtin_ia32_vbroadcastss(XY));

	long l = 0;
	a00 = *((__v2di*)(in               + l));
	a01 = *((__v2di*)(in         + L   + l));
	a02 = *((__v2di*)(in         + 2*L + l));
	a10 = *((__v2di*)(in + X*L         + l));
	a11 = *((__v2di*)(in + X*L   + L   + l));
	a12 = *((__v2di*)(in + X*L   + 2*L + l));
	a20 = *((__v2di*)(in + 2*X*L       + l));
	a21 = *((__v2di*)(in + 2*X*L + L   + l));
	a22 = *((__v2di*)(in + 2*X*L + 2*L + l));

	for(l = 0; l < L - 4; l+=4) {

		b0 = __builtin_e2k_qpfadds(a00, a01);
		b1 = __builtin_e2k_qpfadds(a02, a10);
		b2 = __builtin_e2k_qpfadds(a11, a12);
		b3 = __builtin_e2k_qpfadds(a20, a21);
		b4 = __builtin_e2k_qpfadds(a22, b0);
		b5 = __builtin_e2k_qpfadds(b1, b2);
		b6 = __builtin_e2k_qpfadds(b3, b4);
		b7 = __builtin_e2k_qpfadds(b5, b6);

		*((__v2di*)(out + l)) = __builtin_e2k_qpfmuls(b7, dXY0);

		a00 = *((__v2di*)(in               + l + 4));
		a01 = *((__v2di*)(in         + L   + l + 4));
		a02 = *((__v2di*)(in         + 2*L + l + 4));
		a10 = *((__v2di*)(in + X*L         + l + 4));
		a11 = *((__v2di*)(in + X*L   + L   + l + 4));
		a12 = *((__v2di*)(in + X*L   + 2*L + l + 4));
		a20 = *((__v2di*)(in + 2*X*L       + l + 4));
		a21 = *((__v2di*)(in + 2*X*L + L   + l + 4));
		a22 = *((__v2di*)(in + 2*X*L + 2*L + l + 4));
	}

	b0 = __builtin_e2k_qpfadds(a00, a01);
	b1 = __builtin_e2k_qpfadds(a02, a10);
	b2 = __builtin_e2k_qpfadds(a11, a12);
	b3 = __builtin_e2k_qpfadds(a20, a21);
	b4 = __builtin_e2k_qpfadds(a22, b0);
	b5 = __builtin_e2k_qpfadds(b1, b2);
	b6 = __builtin_e2k_qpfadds(b3, b4);
	b7 = __builtin_e2k_qpfadds(b5, b6);

	*((__v2di*)(out + l))  = __builtin_e2k_qpfmuls(b7, dXY0);

}

void pool_avg_global_core55(float *__restrict__ in, float *__restrict__ out, float* XY, long X, long L, long LL) {

	__v2di a00, a01, a02, a03, a04,
		   a10, a11, a12, a13, a14,
		   a20, a21, a22, a23, a24,
		   a30, a31, a32, a33, a34,
		   a40, a41, a42, a43, a44;

	__v2di b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23;

	__v2di c0, c1;

	__v2di dXY0 = (__v2di)(__builtin_ia32_vbroadcastss(XY));
	// dXY0 = (__v2di)__builtin_ia32_rcpps((__v4sf)(dXY0)); // 1 / (X*Y)

	long l = 0;

	for(l = 0; l < L; l+=4) {

		// printf("Ll = %d\n", l);
		a00 = *((__v2di*)(in               + l));
		a01 = *((__v2di*)(in         + LL   + l));
		a02 = *((__v2di*)(in         + 2*LL + l));
		a03 = *((__v2di*)(in         + 3*LL + l));
		a04 = *((__v2di*)(in         + 4*LL + l));

		a10 = *((__v2di*)(in + X*LL          + l));
		a11 = *((__v2di*)(in + X*LL   + LL   + l));
		a12 = *((__v2di*)(in + X*LL   + 2*LL + l));
		a13 = *((__v2di*)(in + X*LL   + 3*LL + l));
		a14 = *((__v2di*)(in + X*LL   + 4*LL + l));

		a20 = *((__v2di*)(in + 2*X*LL         + l));
		a21 = *((__v2di*)(in + 2*X*LL +  LL   + l));
		a22 = *((__v2di*)(in + 2*X*LL +  2*LL + l));
		a23 = *((__v2di*)(in + 2*X*LL +  3*LL + l));
		a24 = *((__v2di*)(in + 2*X*LL +  4*LL + l));

		a30 = *((__v2di*)(in + 3*X*LL        + l));
		a31 = *((__v2di*)(in + 3*X*LL + LL   + l));
		a32 = *((__v2di*)(in + 3*X*LL + 2*LL + l));
		a33 = *((__v2di*)(in + 3*X*LL + 3*LL + l));
		a34 = *((__v2di*)(in + 3*X*LL + 4*LL + l));

		a40 = *((__v2di*)(in + 4*X*LL        + l));
		a41 = *((__v2di*)(in + 4*X*LL + LL   + l));
		a42 = *((__v2di*)(in + 4*X*LL + 2*LL + l));
		a43 = *((__v2di*)(in + 4*X*LL + 3*LL + l));
		a44 = *((__v2di*)(in + 4*X*LL + 4*LL + l));

		b0  = __builtin_e2k_qpfadds(a00, a01);
		b1  = __builtin_e2k_qpfadds(a02, a03);
		b2  = __builtin_e2k_qpfadds(a04, a10);
		b3  = __builtin_e2k_qpfadds(a11, a12);
		b4  = __builtin_e2k_qpfadds(a13, a14);
		b5  = __builtin_e2k_qpfadds(a20, a21);
		b6  = __builtin_e2k_qpfadds(a22, a23);
		b7  = __builtin_e2k_qpfadds(a24, a30);
		b8  = __builtin_e2k_qpfadds(a31, a32);
		b9  = __builtin_e2k_qpfadds(a33, a34);
		b10 = __builtin_e2k_qpfadds(a40, a41);
		b11 = __builtin_e2k_qpfadds(a42, a43);
		b12 = __builtin_e2k_qpfadds(a44, b0);

		b13 = __builtin_e2k_qpfadds(b1, b2);
		b14 = __builtin_e2k_qpfadds(b3, b4);
		b15 = __builtin_e2k_qpfadds(b5, b6);
		b16 = __builtin_e2k_qpfadds(b7, b8);
		b17 = __builtin_e2k_qpfadds(b9, b10);
		b18 = __builtin_e2k_qpfadds(b11, b12);

		b19 = __builtin_e2k_qpfadds(b13, b14);
		b20 = __builtin_e2k_qpfadds(b15, b16);
		b21 = __builtin_e2k_qpfadds(b17, b18);

		b22 = __builtin_e2k_qpfadds(b19, b20);

		b23 = __builtin_e2k_qpfadds(b21, b22);


		// c0 = *((__v2di*)(out + l));
		// c1 = __builtin_e2k_qpfadds(b23, c0);

		// *((__v2di*)(out + l)) = b23;
		*((__v2di*)(out + l)) = __builtin_e2k_qpfmuls(b23, dXY0);

	}

}

void pool_avg_global_core73(float *__restrict__ in, float *__restrict__ out, float *XY, long X, long L) {

	__v2di a00, a01, a02, a03, a04, a05, a06,
		   a10, a11, a12, a13, a14, a15, a16,
		   a20, a21, a22, a23, a24, a25, a26;

	__v2di b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25, b26;

	__v2di c0, c1;

	__v2di dXY0 = (__v2di)(__builtin_ia32_vbroadcastss(XY));
	// dXY0 = (__v2di)__builtin_ia32_rcpps((__v4sf)(dXY0)); // 1 / (X*Y)

	long l = 0;

	for(l = 0; l < L; l+=4) {

		// printf("Ll = %d\n", l);
		a00 = *((__v2di*)(in                 + l));
		a01 = *((__v2di*)(in           + L   + l));
		a02 = *((__v2di*)(in           + 2*L + l));
		a03 = *((__v2di*)(in           + 3*L + l));
		a04 = *((__v2di*)(in           + 4*L + l));
		a05 = *((__v2di*)(in           + 5*L + l));
		a06 = *((__v2di*)(in           + 6*L + l));

		a10 = *((__v2di*)(in + X*L           + l));
		a11 = *((__v2di*)(in + X*L     + L   + l));
		a12 = *((__v2di*)(in + X*L     + 2*L + l));
		a13 = *((__v2di*)(in + X*L     + 3*L + l));
		a14 = *((__v2di*)(in + X*L     + 4*L + l));
		a15 = *((__v2di*)(in + X*L     + 5*L + l));
		a16 = *((__v2di*)(in + X*L     + 6*L + l));

		a20 = *((__v2di*)(in + 2*X*L         + l));
		a21 = *((__v2di*)(in + 2*X*L   + L   + l));
		a22 = *((__v2di*)(in + 2*X*L   + 2*L + l));
		a23 = *((__v2di*)(in + 2*X*L   + 3*L + l));
		a24 = *((__v2di*)(in + 2*X*L   + 4*L + l));
		a25 = *((__v2di*)(in + 2*X*L   + 5*L + l));
		a26 = *((__v2di*)(in + 2*X*L   + 6*L + l));

		b0 = __builtin_e2k_qpfadds(a00, a01);
		b1 = __builtin_e2k_qpfadds(a02, a03);
		b2 = __builtin_e2k_qpfadds(a04, a05);
		b3 = __builtin_e2k_qpfadds(a06, a10);
		b4 = __builtin_e2k_qpfadds(a11, a12);
		b5 = __builtin_e2k_qpfadds(a13, a14);
		b6 = __builtin_e2k_qpfadds(a15, a16);
		b7 = __builtin_e2k_qpfadds(a20, a21);
		b8 = __builtin_e2k_qpfadds(a22, a23);
		b9 = __builtin_e2k_qpfadds(a24, a25);

		b10 = __builtin_e2k_qpfadds(a26, b0);
		b11 = __builtin_e2k_qpfadds(b1,  b2);
		b12 = __builtin_e2k_qpfadds(b3,  b4);
		b13 = __builtin_e2k_qpfadds(b5,  b6);
		b14 = __builtin_e2k_qpfadds(b7,  b8);

		b15 = __builtin_e2k_qpfadds(b9,  b10);
		b16 = __builtin_e2k_qpfadds(b11, b12);
		b17 = __builtin_e2k_qpfadds(b13, b14);

		b18 = __builtin_e2k_qpfadds(b15, b16);

		b19 = __builtin_e2k_qpfadds(b17, b18);


		c0 = *((__v2di*)(out + l));
		c1 = __builtin_e2k_qpfadds(b19, c0);

		*((__v2di*)(out + l)) = __builtin_e2k_qpfmuls(c1, dXY0);


	}

}

void pool_avg_global_core74(float *__restrict__ in, float *__restrict__ out, long X, long L) {

	__v2di a00, a01, a02, a03, a04, a05, a06,
		   a10, a11, a12, a13, a14, a15, a16,
		   a20, a21, a22, a23, a24, a25, a26,
		   a30, a31, a32, a33, a34, a35, a36;

	__v2di b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25, b26;

	__v2di c0, c1;

	long l = 0;

	for(l = 0; l < L; l+=4) {

		a00 = *((__v2di*)(in                 + l));
		a01 = *((__v2di*)(in           + L   + l));
		a02 = *((__v2di*)(in           + 2*L + l));
		a03 = *((__v2di*)(in           + 3*L + l));
		a04 = *((__v2di*)(in           + 4*L + l));
		a05 = *((__v2di*)(in           + 5*L + l));
		a06 = *((__v2di*)(in           + 6*L + l));

		a10 = *((__v2di*)(in + X*L           + l));
		a11 = *((__v2di*)(in + X*L     + L   + l));
		a12 = *((__v2di*)(in + X*L     + 2*L + l));
		a13 = *((__v2di*)(in + X*L     + 3*L + l));
		a14 = *((__v2di*)(in + X*L     + 4*L + l));
		a15 = *((__v2di*)(in + X*L     + 5*L + l));
		a16 = *((__v2di*)(in + X*L     + 6*L + l));

		a20 = *((__v2di*)(in + 2*X*L         + l));
		a21 = *((__v2di*)(in + 2*X*L   + L   + l));
		a22 = *((__v2di*)(in + 2*X*L   + 2*L + l));
		a23 = *((__v2di*)(in + 2*X*L   + 3*L + l));
		a24 = *((__v2di*)(in + 2*X*L   + 4*L + l));
		a25 = *((__v2di*)(in + 2*X*L   + 5*L + l));
		a26 = *((__v2di*)(in + 2*X*L   + 6*L + l));

		a30 = *((__v2di*)(in + 3*X*L         + l));
		a31 = *((__v2di*)(in + 3*X*L   + L   + l));
		a32 = *((__v2di*)(in + 3*X*L   + 2*L + l));
		a33 = *((__v2di*)(in + 3*X*L   + 3*L + l));
		a34 = *((__v2di*)(in + 3*X*L   + 4*L + l));
		a35 = *((__v2di*)(in + 3*X*L   + 5*L + l));
		a36 = *((__v2di*)(in + 3*X*L   + 6*L + l));

		b0 = __builtin_e2k_qpfadds(a00, a01);
		b1 = __builtin_e2k_qpfadds(a02, a03);
		b2 = __builtin_e2k_qpfadds(a04, a05);
		b3 = __builtin_e2k_qpfadds(a06, a10);
		b4 = __builtin_e2k_qpfadds(a11, a12);
		b5 = __builtin_e2k_qpfadds(a13, a14);
		b6 = __builtin_e2k_qpfadds(a15, a16);
		b7 = __builtin_e2k_qpfadds(a20, a21);
		b8 = __builtin_e2k_qpfadds(a22, a23);
		b9 = __builtin_e2k_qpfadds(a24, a25);
		b10 = __builtin_e2k_qpfadds(a26, a30);
		b11 = __builtin_e2k_qpfadds(a31, a32);
		b12 = __builtin_e2k_qpfadds(a33, a34);
		b13 = __builtin_e2k_qpfadds(a35, a36);


		b14 = __builtin_e2k_qpfadds(b0,  b1);
		b15 = __builtin_e2k_qpfadds(b2,  b3);
		b16 = __builtin_e2k_qpfadds(b4,  b5);
		b17 = __builtin_e2k_qpfadds(b6,  b7);
		b18 = __builtin_e2k_qpfadds(b8,  b9);
		b19 = __builtin_e2k_qpfadds(b10, b11);
		b20 = __builtin_e2k_qpfadds(b12, b13);
		
		b21 = __builtin_e2k_qpfadds(b14, b15);
		b22 = __builtin_e2k_qpfadds(b16, b17);
		b23 = __builtin_e2k_qpfadds(b18, b19);

		b24 = __builtin_e2k_qpfadds(b20, b21);
		b25 = __builtin_e2k_qpfadds(b22, b23);

		b26 = __builtin_e2k_qpfadds(b24, b25);


		// c0 = *((__v2di*)(out + l));
		// c1 = __builtin_e2k_qpfadds(b26, c0);

		// *((__v2di*)(out + l)) = c1;

		*((__v2di*)(out + l)) = b26;

	}

}

void pool_avg_global(float *__restrict__ in, float *__restrict__ out, float *__restrict__  XY, long B, long X, long Y, long L) {

	*XY = 1./(X*Y);
	
	bool fl = true;
	#ifndef _OPENMP // пока оставляем только последовательную версию
		if(X == Y) {
			if(X == 3) {
				for(int b = 0; b < B; ++b) 
					pool_avg_global_core33(in + b*Y*X*L, out + b*L, XY, X, L);
				fl = false;
			}
			else if(X == 5) {
				for(int b = 0; b < B; ++b) 
					pool_avg_global_core55(in + b*Y*X*L, out + b*L, XY, X, L, L);
				fl = false;
			}
			else if(X == 7) {
				for(int b = 0; b < B; ++b) {
					pool_avg_global_core74(in + b*Y*X*L,        out + b*L, X, L);
					pool_avg_global_core73(in + b*Y*X*L + 28*L, out + b*L, XY, X, L);
				}
				fl = false;
			}
		}
		if(fl) {
			long x, y, xy;
			for(int b = 0; b < B; ++b) {
				x = 0; y = 0;
				pool_avg_global_core11_init(in + b*Y*X*L + y*X*L + x*L, out + b*L, L); 
				for(int xy = 1; xy < X*Y - 1; ++xy) {
					x = xy % X;
					y = xy / X;
					pool_avg_global_core11(in + b*Y*X*L + y*X*L + x*L, out + b*L, L); 
				}	
				x++;
				pool_avg_global_core11_last(in + b*Y*X*L + y*X*L + x*L, out + b*L, XY, L); 
			}
		}
	#else	
		int np = 1;
		if(3827.62 - 322.857*X < L) np = 2;

		omp_set_num_threads(np);
		if(X == Y) {
			if(X == 3) {
				#pragma omp parallel
				{	
					int myid = omp_get_thread_num();
					for(long b = myid; b < B; b+=np) 
						pool_avg_global_core33(in + b*Y*X*L, out + b*L, XY, X, L);
					fl = false;
				}
			}
			else if(X == 5) {
				#pragma omp parallel
				{	

					int myid = omp_get_thread_num();
					for(long b = myid; b < B; b+=np) {
						// pool_avg_global_core55(in + b*Y*X*L + myid*(L>>1), out + b*L + myid*(L>>1), XY, X, L>>1, L); // попытка параллелить по L
						pool_avg_global_core55(in + b*Y*X*L, out + b*L, XY, X, L, L); // попытка параллелить по B
					}
					fl = false;
				}
			}
			else if(X == 7) {
				#pragma omp parallel
				{	
					int myid = omp_get_thread_num();
					for(long b = myid; b < B; b+=np) {
						pool_avg_global_core74(in + b*Y*X*L,        out + b*L, X, L);
						pool_avg_global_core73(in + b*Y*X*L + 28*L, out + b*L, XY, X, L);
					}
					fl = false;
					
				}
			}
		}
		if(fl) {
			#pragma omp parallel
			{	
				long x, y, xy;
				int myid = omp_get_thread_num();
				for(long b = myid; b < B; b+=np) {
					x = 0; y = 0;
					pool_avg_global_core11_init(in + b*Y*X*L + y*X*L + x*L, out + b*L, L); 
					for(int xy = 1; xy < X*Y - 1; ++xy) {
						x = xy % X;
						y = xy / X;
						pool_avg_global_core11(in + b*Y*X*L + y*X*L + x*L, out + b*L, L); 
					}	
					x++;
					pool_avg_global_core11_last(in + b*Y*X*L + y*X*L + x*L, out + b*L, XY, L); 
				}
			}
		}	
	#endif


}

