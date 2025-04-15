#ifndef __Bcpnn_Kernel_included
#define __Bcpnn_Kernel_included

#include <vector>
#include <cstring>

#include "Globals.h"

#define ABSENT 0
#define SILENT 1
#define ACTIVE 2

#define UNSUPERVISED 0
#define SUPERVISED 1
#define INFERENCES 2


const int NumberPop = 3;
const float eps_hls = 1e-8;
const float eps_neg_hls = -1e-8;
const float EPS_GLB = 1e-7;

// Layer Population Input
const int H_in = 784;
const int M_in = 2;
const int N_in = H_in * M_in;

// Layer Population Hidden
const int H_hid = 32;
const int M_hid = 128;
const int N_hid = H_hid * M_hid;
const float M_hid_inv = 1.0f/M_hid;
const int log2M_hid = 7;

// Layer Population Output
const int H_ut = 1;
const int M_ut = 10;
const int N_ut = H_ut * M_ut;

const int nactHi_pop = 64;
const int nsilHi_pop = 64;
const float fgain = 1.0;
const float tauzjdt = 1.0;
const float tauzjdt_neg = 0.0;
const float tauzidt = 1.0;
const float again_hls = 1.0;

// Layer Projection Input to Hidden
const int axoHi_ih = H_in;
const int axoNi_ih = axoHi_ih*M_in;
const int nactHi_ih = nactHi_pop;
const int denHi_ih = nactHi_ih + nsilHi_pop;
const int denNi_ih = denHi_ih*M_in;
const int denNi_ih_log2 = 8;

// Layer Projection Hidden to Output
const int axoHi_hu = H_hid;
const int axoNi_hu = axoHi_hu*M_hid;
const int nactHi_hu = axoHi_hu;
const int denHi_hu = nactHi_hu + 0;
const int denNi_hu = denHi_hu*M_hid;

const float eps_hls_m_tauzjdt = eps_hls*tauzjdt;
const float eps_hls_m_tauzjdt_neg = -eps_hls_m_tauzjdt;
const float eps_hls_m_tauzidt = eps_hls*tauzidt;
const float eps_hls_m_tauzidt_neg = -eps_hls_m_tauzidt;

const float tauzidt_neg = (1-tauzidt);

const int H_hid__denHi_ih = H_hid*denHi_ih;
const int H_hid__denNi_ih = H_hid*denNi_ih;
const int N_hid__denNi_ih = N_hid*denNi_ih;
const int H_ut__denNi_hu = H_ut*denNi_hu;
const int N_ut__denNi_hu = N_ut*denNi_hu;

const float eps_hls_tauzidt = eps_hls*tauzidt;
const float eps_neg_tauzidt = 1 - eps_hls_tauzidt;


// enum for modeOps
enum modeOps {
    UNSUPERVISED_TRAIN = 0,
    SUPERVISED_TRAIN = 1,
    INFERENCES_MODE = 2
};

//void BCPNN_Kernel(float *input_hbm, float *label_hbm, float *output_hbm, int modeOps, float *rndPoisson_hid_hbm, int *Hihjhi_ih_hbm, int *Chjhi_ih_hbm, bool *needsupdbw_hbm,
//                  float *Zj_ih_hbm, float *Zi_ih_hbm, float *Pj_ih_hbm, float *Pi_ih_hbm, float *Pji_ih_hbm, float *Bj_ih_hbm, float *Wji_ih_hbm,
//                  float *Zj_hu_hbm, float *Zi_hu_hbm, float *Pj_hu_hbm, float *Pi_hu_hbm, float *Pji_hu_hbm, float *Bj_hu_hbm, float *Wji_hu_hbm,
//                  float *constant_hbm);
//void BCPNN_Kernel(float *input_hbm, float *label_hbm, float *output_hbm, int modeOps, float *rndPoisson_hid_hbm, int *Hihjhi_ih_hbm, int *Chjhi_ih_hbm, bool *needsupdbw_hbm,
//                  float *Pj_ih_hbm, float *Pi_ih_hbm, float *Pji_ih_hbm, float *Bj_ih_hbm, float *Wji_ih_hbm,  
//                  float *Pj_hu_hbm, float *Pi_hu_hbm, float *Pji_hu_hbm, float *Bj_hu_hbm, float *Wji_hu_hbm,
//                  float *constant_hbm);
void BCPNN_Kernel(float *input_hbm, float *label_hbm, float *output_hbm, int modeOps, float *rndPoisson_hid_hbm, int *Hihjhi_ih_hbm, int *Chjhi_ih_hbm, char *needsupdbw_hbm,
                  float *Zj_ih_hbm, float *Zi_ih_hbm, float *Pj_ih_hbm, float *Pi_ih_hbm, float *Pji_ih_hbm, float *Bj_ih_hbm, float *Wji_ih_hbm,
                  float *Zj_hu_hbm, float *Zi_hu_hbm, float *Pj_hu_hbm, float *Pi_hu_hbm, float *Pji_hu_hbm, float *Bj_hu_hbm, float *Wji_hu_hbm,
                  float *constant_hbm);
void BCPNN_Inference(float *input_hbm, float *label_hbm, float *output_hbm, int modeOps, float *rndPoisson_hid_hbm, int *Hihjhi_ih_hbm, int *Chjhi_ih_hbm, char *needsupdbw_hbm,
                  float *Pj_ih_hbm, float *Pi_ih_hbm, float *Pji_ih_hbm, float *Pji_ih_hbm1, float *Pji_ih_hbm2, float *Pji_ih_hbm3,   
                  float *Bj_ih_hbm, float *Wji_ih_hbm, float *Wji_ih_hbm1, float *Wji_ih_hbm2, float *Wji_ih_hbm3,
                  float *Pj_hu_hbm, float *Pi_hu_hbm, float *Pji_hu_hbm, float *Bj_hu_hbm, float *Wji_hu_hbm,
                  float *constant_hbm);
//void BCPNN_Kernel(float *input_hbm, float *label_hbm, float *output_hbm, int modeOps, float *rndPoisson_hid_hbm, int *Hihjhi_ih_hbm, int *Chjhi_ih_hbm, bool *needsupdbw_hbm,
//                  float *Pj_ih_hbm, float *Pi_ih_hbm, float *Pji_ih_hbm, float *Pji_ih_hbm1, float *Pji_ih_hbm2, float *Pji_ih_hbm3,
//                  float *Pji_ih_hbm4, float *Pji_ih_hbm5, float *Pji_ih_hbm6, float *Pji_ih_hbm7,
//                  float *Bj_ih_hbm, float *Wji_ih_hbm, float *Wji_ih_hbm1, float *Wji_ih_hbm2, float *Wji_ih_hbm3,
//                  float *Wji_ih_hbm4, float *Wji_ih_hbm5, float *Wji_ih_hbm6, float *Wji_ih_hbm7,
//                  float *Pj_hu_hbm, float *Pi_hu_hbm, float *Pji_hu_hbm, float *Bj_hu_hbm, float *Wji_hu_hbm,
//                  float *constant_hbm);

#endif // __Bcpnn_Kernel_included