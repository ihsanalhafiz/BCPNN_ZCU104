/*

  Author: Anders Lansner

  Created: 2024-07-30     Modified: 2024-07-30
  Based on former ../../tests/prjhtest3.cpp via ../../apps/MNIST

*/

#include <vector>
#include <string>
#include <sys/time.h>

#include "Globals.h"
#include "Pop.h"
#include "Prj.h"
#include "Logger.h"
#include "Probe.h"
#include "Parseparam.h"
#include "PatternFactory.h"
#include "Analys.h"
#include "BCPNN_Kernel.h"

using namespace std;
using namespace Globals;
using namespace Logging;
using namespace Probing;

long seed = 0;
int Hx = 5, Mx = 5, Nx, Hin = Hx, Min = Mx, Nin, Hhid = Hin, Mhid = Min, Hut = 1, Mut = 10, Nut, nhlayer = 1,
    updconnint = -1;
bool inbinarize = false, doshuffle = true, logenabled = true;
float eps = -1, maxfq = 1. / timestep, taup = -1, bgain = 1, wgain = 1, again = 1, cthres = 0.1, nampl = 0,
      nfreq = 1, nswap = 0, nrepl = 0, swaprthr = 1.1, replrthr = 1, replthr = -1;
string lrule = "BCP", actfn = "WTA", patype = "hrand";
string trainpatfile = "", trainlblfile = "", testpatfile = "", testlblfile = "";
int nactHi = 0, nsilHi = 0, nrep = 1, nepoch = 1,
    trnpat = 10, tenpat = -1, nloop = -1, nhblank = 0, nhflip = 0, myverbosity = 1;
PatternFactory *trainpatfac = nullptr, *trainlblfac = nullptr, *testpatfac = nullptr,
                *testlblfac = nullptr;
Pop *inpop, *hidpop, *utpop;
Prj *ihprj, *huprj;

float *rndPoisson_in, *rndPoisson_hid, *rndPoisson_ut;
int *Hihjhi_ih, *Chjhi_ih;
float *Zj_ih, *Zi_ih, *Pj_ih, *Pi_ih, *Pji_ih, *Pji_ih1, *Pji_ih2, *Pji_ih3, *Wji_ih, *Wji_ih1, *Wji_ih2, *Wji_ih3, *Bj_ih;
float *Zj_hu, *Zi_hu, *Pj_hu, *Pi_hu, *Pji_hu, *Wji_hu, *Bj_hu;
char *needsupdbw;
float bgain_ih = 1, wgain_ih = 1, ewgain_ih = 1, iwgain_ih = 1;
float bgain_hu = 1, wgain_hu = 1, ewgain_hu = 1, iwgain_hu = 1;
float igain_pop[NumberPop] = {1, 1, 1};
float bwgain_pop[NumberPop] = {1, 1, 1};
float nampl_pop = 0;
float nfreq_pop = 1;
float taumdt_pop[NumberPop] = {1, 1, 1};
float prntaudpt[2] = {0, 0};
float *constant_hbm;

struct timeval total_time;

void allocVarKernel() {
    rndPoisson_in = new float[N_in];
    rndPoisson_hid = new float[N_hid];
    rndPoisson_ut = new float[N_ut];
    Hihjhi_ih = new int[H_hid*denHi_ih];
    Chjhi_ih = new int[H_hid*denHi_ih];
    Zj_ih = new float[N_hid];
    Zi_ih = new float[H_hid * denNi_ih];
    Pj_ih = new float[N_hid];
    Pi_ih = new float[H_hid * denNi_ih];
    Pji_ih = new float[(N_hid*denNi_ih)];
    Wji_ih = new float[(N_hid*denNi_ih)];
    Bj_ih = new float[N_hid];
    Zj_hu = new float[N_ut];
    Zi_hu = new float[H_ut * denNi_hu];
    Pj_hu = new float[N_ut];
    Pi_hu = new float[H_ut * denNi_hu];
    Pji_hu = new float[N_ut * denNi_hu];
    Wji_hu = new float[N_ut * denNi_hu];
    Bj_hu = new float[N_ut];
    needsupdbw = new char[2];
    constant_hbm = new float[21];
}

void copyDataToBuffer(Pop *inpop, Pop *hidpop, Pop *utpop, Prj *ihprj, Prj *huprj,
                      float *rndPoisson_in, float *rndPoisson_hid, float *rndPoisson_ut,
                      int *Hihjhi_ih, int *Chjhi_ih, 
                      float *Zj_ih, float *Zi_ih, float *Pj_ih, float *Pi_ih, 
                      float *Pji_ih, float *Wji_ih, float *Bj_ih,
                      float *Zj_hu, float *Zi_hu, float *Pj_hu, float *Pi_hu, float *Pji_hu, float *Wji_hu, float *Bj_hu,
                      char *needsupdbw, float *constant_hbm, int modeOps) {
    constant_hbm[0] = ihprj->bgain;
    constant_hbm[1] = ihprj->wgain;
    constant_hbm[2] = ihprj->ewgain;
    constant_hbm[3] = ihprj->iwgain;
    constant_hbm[4] = huprj->bgain;
    constant_hbm[5] = huprj->wgain;
    constant_hbm[6] = huprj->ewgain;
    constant_hbm[7] = huprj->iwgain;
    constant_hbm[8] = hidpop->nampl;
    constant_hbm[9] = hidpop->nfreq;
    if(modeOps == UNSUPERVISED) constant_hbm[10] = ihprj->taupdt;
    else constant_hbm[10] = 0.0;
    if(modeOps == SUPERVISED) constant_hbm[11] = huprj->taupdt;
    else constant_hbm[11] = 0.0;
    constant_hbm[12] = inpop->igain;
    constant_hbm[13] = hidpop->igain;
    constant_hbm[14] = utpop->igain;
    constant_hbm[15] = inpop->bwgain;
    constant_hbm[16] = hidpop->bwgain;
    constant_hbm[17] = utpop->bwgain;
    constant_hbm[18] = inpop->taumdt;
    constant_hbm[19] = hidpop->taumdt;
    constant_hbm[20] = utpop->taumdt;
    needsupdbw[0] = ihprj->needsupdbw;
    needsupdbw[1] = huprj->needsupdbw;
    gsetpoissonmean(inpop->nfreq);
    for (int i = 0; i < N_in; i++)
        rndPoisson_in[i] = gnextpoisson();
    gsetpoissonmean(hidpop->nfreq);
    for (int i = 0; i < N_hid; i++)
        rndPoisson_hid[i] = gnextpoisson();
    gsetpoissonmean(utpop->nfreq);
    for (int i = 0; i < N_ut; i++)
        rndPoisson_ut[i] = gnextpoisson();
    
    memcpy(Hihjhi_ih, ihprj->Hihjhi, H_hid*denHi_ih*sizeof(int));
    memcpy(Chjhi_ih, ihprj->Chjhi, H_hid*denHi_ih*sizeof(int));
    memcpy(Zj_ih, ihprj->Zj, N_hid*sizeof(float));
    memcpy(Zi_ih, ihprj->Zi, H_hid*denNi_ih*sizeof(float));
    memcpy(Pj_ih, ihprj->Pj, N_hid*sizeof(float));
    memcpy(Pi_ih, ihprj->Pi, H_hid*denNi_ih*sizeof(float));
    //memcpy(Pji_ih, ihprj->Pji, N_hid*denNi_ih*sizeof(float));
    //memcpy(Wji_ih, ihprj->Wji, N_hid*denNi_ih*sizeof(float));
    for(int i = 0; i < (N_hid*denNi_ih); i++){
            Pji_ih[i] = ihprj->Pji[i];
            Wji_ih[i] = ihprj->Wji[i];        
    }
    memcpy(Bj_ih, ihprj->Bj, N_hid*sizeof(float));
    memcpy(Zj_hu, huprj->Zj, N_ut*sizeof(float));
    memcpy(Zi_hu, huprj->Zi, H_ut*denNi_hu*sizeof(float));
    memcpy(Pj_hu, huprj->Pj, N_ut*sizeof(float));
    memcpy(Pi_hu, huprj->Pi, H_ut*denNi_hu*sizeof(float));
    memcpy(Pji_hu, huprj->Pji, N_ut*denNi_hu*sizeof(float));
    memcpy(Wji_hu, huprj->Wji, N_ut*denNi_hu*sizeof(float));
    memcpy(Bj_hu, huprj->Bj, N_ut*sizeof(float));
}
void copyBufToData(Prj *ihprj, Prj *huprj, char *needsupdbw,
                   float *Zj_ih, float *Zi_ih, float *Pj_ih, float *Pi_ih, 
                   float *Pji_ih, float *Wji_ih, float *Bj_ih,
                   float *Zj_hu, float *Zi_hu, float *Pj_hu, float *Pi_hu, float *Pji_hu, float *Wji_hu, float *Bj_hu) {
    memcpy(ihprj->Zj, Zj_ih, N_hid*sizeof(float));
    memcpy(ihprj->Zi, Zi_ih, H_hid*denNi_ih*sizeof(float));
    memcpy(ihprj->Pj, Pj_ih, N_hid*sizeof(float));
    memcpy(ihprj->Pi, Pi_ih, H_hid*denNi_ih*sizeof(float));
    //memcpy(ihprj->Pji, Pji_ih, N_hid*denNi_ih*sizeof(float));
    //memcpy(ihprj->Wji, Wji_ih, N_hid*denNi_ih*sizeof(float));
    for(int i = 0; i < (N_hid*denNi_ih); i++){
            ihprj->Pji[i] = Pji_ih[i];
            ihprj->Wji[i] = Wji_ih[i];
    }
    memcpy(ihprj->Bj, Bj_ih, N_hid*sizeof(float));
    memcpy(huprj->Zj, Zj_hu, N_ut*sizeof(float));
    memcpy(huprj->Zi, Zi_hu, H_ut*denNi_hu*sizeof(float));
    memcpy(huprj->Pj, Pj_hu, N_ut*sizeof(float));
    memcpy(huprj->Pi, Pi_hu, H_ut*denNi_hu*sizeof(float));
    memcpy(huprj->Pji, Pji_hu, N_ut*denNi_hu*sizeof(float));
    memcpy(huprj->Wji, Wji_hu, N_ut*denNi_hu*sizeof(float));
    memcpy(huprj->Bj, Bj_hu, N_ut*sizeof(float));
    ihprj->needsupdbw = needsupdbw[0];
    huprj->needsupdbw = needsupdbw[1];
}

void parseparams(std::string paramfile) {
    Parseparam *parseparam = new Parseparam(paramfile);
    parseparam->postparam("seed", &seed, Long);
    parseparam->postparam("Hx", &Hx, Int);
    parseparam->postparam("Mx", &Mx, Int);
    parseparam->postparam("Hin", &Hin, Int);
    parseparam->postparam("Min", &Min, Int);
    parseparam->postparam("Hhid", &Hhid, Int);
    parseparam->postparam("Mhid", &Mhid, Int);
    parseparam->postparam("Hut", &Hut, Int);
    parseparam->postparam("Mut", &Mut, Int);
    parseparam->postparam("lrule", &lrule, String);
    parseparam->postparam("actfn", &actfn, String);
    parseparam->postparam("maxfq", &maxfq, Float);
    parseparam->postparam("nactHi", &nactHi, Int);
    parseparam->postparam("nsilHi", &nsilHi, Int);
    parseparam->postparam("nfreq", &nfreq, Float);
    parseparam->postparam("nampl", &nampl, Float);
    parseparam->postparam("again", &again, Float);
    parseparam->postparam("eps", &eps, Float);
    parseparam->postparam("taup", &taup, Float);
    parseparam->postparam("bgain", &bgain, Float);
    parseparam->postparam("wgain", &wgain, Float);
    parseparam->postparam("swaprthr", &swaprthr, Float);
    parseparam->postparam("replrthr", &replrthr, Float);
    parseparam->postparam("replthr", &replthr, Float);
    parseparam->postparam("updconnint", &updconnint, Int);
    parseparam->postparam("inbinarize", &inbinarize, Boole);
    parseparam->postparam("trainpatfile", &trainpatfile, String);
    parseparam->postparam("trainlblfile", &trainlblfile, String);
    parseparam->postparam("testpatfile", &testpatfile, String);
    parseparam->postparam("testlblfile", &testlblfile, String);
    parseparam->postparam("nrep", &nrep, Int);
    parseparam->postparam("patype", &patype, String);
    parseparam->postparam("nswap", &nswap, Float);
    parseparam->postparam("nrepl", &nrepl, Float);
    parseparam->postparam("nepoch", &nepoch, Int);
    parseparam->postparam("trnpat", &trnpat, Int);
    parseparam->postparam("tenpat", &tenpat, Int);
    parseparam->postparam("nloop", &nloop, Int);
    parseparam->postparam("nhblank", &nhblank, Int);
    parseparam->postparam("nhflip", &nhflip, Int);
    parseparam->postparam("doshuffle", &doshuffle, Boole);
    parseparam->postparam("cthres", &cthres, Float);
    parseparam->postparam("logenabled", &logenabled, Boole);
    parseparam->postparam("libverbosity", &verbosity, Int);
    parseparam->postparam("verbosity", &myverbosity, Int);
    parseparam->postparam("verbosity", &verbosity, Int);
    parseparam->doparse();

    if (replthr != -1)
        error("mnistmain::parseparam","parameter 'replthr' changed name to 'replrthr'");

    Nx = Hx * Mx;
    Nin = Hin * Min;
    Nut = Hut * Mut;
    if (tenpat < 0)
        tenpat = trnpat;
}

void maketrainpats(string type = "all") {
    if (type == "all" or type == "pats") {
        trainpatfac = new PatternFactory(Nx, Hx, patype);
        trainpatfac->mkpats(trnpat);
    }
    if (type == "all" or type == "lbls") {
        trainlblfac = new PatternFactory(Nut, Hut, patype);
        trainlblfac->mkpats(trnpat);
    }
}

void maketestpats(string type = "all") {
    if (type == "all" or type == "pats") {
        testpatfac = new PatternFactory(trainpatfac);
        testpatfac->hflippats(nhflip);
        testpatfac->hblankpats(nhblank);
    }
    if (type == "all" or type == "lbls") {
        testlblfac = new PatternFactory(trainlblfac);
    }
}

void readtrainpats(string type = "all", bool binarize = false) {
    if (type == "all" or type == "pats") {
        if (Nx == Hx and binarize == false)
            error("main::readtrainpats", "Illegal: Nx == Hx and not binarize");
        trainpatfac = new PatternFactory(Nx, Hx, patype);
        trainpatfac->readpats(trainpatfile, trnpat);
        if (binarize) {
            trainpatfac->binarizepats(trainpatfac->pats);
        }
    }
    if (type == "all" or type == "lbls") {
        trainlblfac = new PatternFactory(Nut, Hut, patype);
        trainlblfac->readpats(trainlblfile, trnpat);
    }
}

void readtestpats(string type = "all", bool binarize = false) {
    if (type == "all" or type == "pats") {
        testpatfac = new PatternFactory(Nx, Hx, patype);
        testpatfac->readpats(testpatfile, tenpat);
        if (binarize)
            testpatfac->binarizepats(testpatfac->pats);
        if (verbosity > 1)
            fprintf(stderr, "nhflip = %d\n", nhflip);
        //testpatfac->hflippats(nhflip);
        //testpatfac->hblankpats(nhblank);
    }
    if (type == "all" or type == "lbls") {
        testlblfac = new PatternFactory(Nut, Hut, patype);
        testlblfac->readpats(testlblfile, tenpat);
    }
}

void makepats(bool binarize = false) {
    if (trainpatfile == "")
        maketrainpats("pats");
    else
        readtrainpats("pats", binarize);
    if (testpatfile == "")
        maketestpats("pats");
    else
        readtestpats("pats", binarize);
    if (trainlblfile == "")
        maketrainpats("lbls");
    else
        readtrainpats("lbls");
    if (testlblfile == "")
        maketestpats("lbls");
    else
        readtestpats("lbls");
}

int main(int argc, char **args) {
    gettimeofday(&total_time, 0);
    string paramfile = "mnistmain.par";
    if (argc > 1)
        paramfile = args[1];
    parseparams(paramfile);

    allocVarKernel();

    float trfcorr = 0, tefcorr = 0;
    vector<float> trfcorrs, tefcorrs;
    Analys *analys1 = new Analys(Hut, Mut, cthres);
    bool pnln = false;
    if (taup < 0) {
        taup = trnpat * timestep / 6.0;
        if (myverbosity > 2) {
            printf("taup = %.3f ", taup);
            fflush(stdout);
        }
        pnln = true;
    }
    if (eps < 0) {
        eps = 1. / (1 + trnpat);
        if (myverbosity > 2) {
            printf("eps = %.1e ", eps);
            fflush(stdout);
        }
        pnln = true;
    }
    if (updconnint < 0) {
        updconnint = 500 * trnpat/60000.0;
        if (myverbosity > 2)
            printf("updconnint = %d", updconnint);
        pnln = true;
    }
    if (myverbosity > 2 and pnln) {
        printf("\n");
        fflush(stdout);
    }
    for (int rep = 0, ncorr; rep < nrep; rep++) {
        if (myverbosity > 2 and nrep > 1)
            fprintf(stderr, "rep = %d ", rep);
        clear();
        gsetseed(seed);
        makepats(inbinarize);
        inpop = new Pop(Hin, Min, "inpop");
        inpop->setactfn(actfn);
        inpop->setmaxfq(maxfq);
        hidpop = new Pop(Hhid, Mhid, "hidpop");
        hidpop->setactfn(actfn);
        hidpop->setnfreq(nfreq);
        hidpop->setnampl(nampl);
        utpop = new Pop(Hut, Mut, "utpop");
        utpop->setactfn(actfn);
        ihprj = new Prj(inpop, hidpop, nactHi, nsilHi, "ihprj");
        ihprj->settaup(taup);
        ihprj->seteps(eps);
        ihprj->setwgain(wgain);
        ihprj->setbgain(bgain);
        ihprj->setnswap(nswap);
        ihprj->setnrepl(nrepl);
        ihprj->setswaprthr(swaprthr);
        ihprj->setreplrthr(replrthr);

        huprj = new Prj(hidpop, utpop, "huprj");
        huprj->settaup(taup);
        huprj->seteps(eps);

        vector<float> prns(Prj::prjs.size());
        fill(prns.begin(), prns.end(), 0);
        // Unsupervised representation learning
        if (verbosity > 1 and nrep == 1) {
            printf("Unsupervised learning (%d)\n", simstep);
            fflush(stdout);
        }
        vector<int> pidx(trnpat);
        for (int p = 0; p < trnpat; p++)
            pidx[p] = p;
        hidpop->setbwgain(1);
        int pdot = max(1, trnpat / 20);

        for (int epoch = 0; epoch < nepoch; epoch++) {
            if (doshuffle)
                pidx = gshuffle(pidx);
            if (nepoch > 1 and myverbosity > 1) {
                if (epoch == 0)
                    fprintf(stderr, "epoch = %2d ", epoch);
                else
                    fprintf(stderr, "%2d ", epoch);
            }
            for (int p = 0, q; p < trnpat; p++) {
                if (myverbosity > 2 and p % pdot == 0)
                    fprintf(stderr, ".");
                q = pidx[p];
                //Pop::resetall();
                //inpop->setinput(trainpatfac->getfpat(q));
                //for (int loop = 0; loop < nloop; loop++) {
                //    if (loop == nloop - 1)
                //        prns[0] = 1;
                //    Pop::updsupall();
                //    Pop::updactall();
                //    Prj::upddenactall();
                //    Prj::updtracesall(prns);
                //    Pop::resetbwsupall();
                //    Prj::updbwsupall();
                //    Prj::contributeall();
                //    Prj::updbwall();
                //    prns[0] = 0;
                //    advance();
                //}
                copyDataToBuffer(inpop, hidpop, utpop, ihprj, huprj,
                                 rndPoisson_in, rndPoisson_hid, rndPoisson_ut,
                                 Hihjhi_ih, Chjhi_ih,
                                 Zj_ih, Zi_ih, Pj_ih, Pi_ih, Pji_ih, Wji_ih, Bj_ih,
                                 Zj_hu, Zi_hu, Pj_hu, Pi_hu, Pji_hu, Wji_hu, Bj_hu,
                                 needsupdbw, constant_hbm, UNSUPERVISED);
                BCPNN_Kernel(trainpatfac->getfpat(q), NULL, utpop->act, UNSUPERVISED, rndPoisson_hid,
                             Hihjhi_ih, Chjhi_ih, needsupdbw,
                             Zj_ih, Zi_ih, Pj_ih, Pi_ih, Pji_ih, Bj_ih, Wji_ih,
                             Zj_hu, Zi_hu, Pj_hu, Pi_hu, Pji_hu, Bj_hu, Wji_hu,
                             constant_hbm);
                copyBufToData(ihprj, huprj, needsupdbw,
                              Zj_ih, Zi_ih, Pj_ih, Pi_ih, Pji_ih, Wji_ih, Bj_ih,
                              Zj_hu, Zi_hu, Pj_hu, Pi_hu, Pji_hu, Wji_hu, Bj_hu);


                if (updconnint > 0 and p%updconnint == 0) {
                    ihprj->swapconns();
                    ihprj->replconns();
                }
            }
            if (updconnint > 0 and myverbosity > 2) {
                if (nswap == 0)
                    ihprj->updMIsc();
                ihprj->miscsum();
                printf("simstep = %d totnswapped = %d totnrepled = %d\n",
                       simstep, ihprj->gnswapped, ihprj->gnrepled);
            } else
                printf("\n");
        }
        hidpop->setbwgain(1);

        // Supervised learning
        if (nrep == 1) {
            printf("Supervised learning (%d)\n", simstep);
            fflush(stdout);
        }
        hidpop->setnampl(0);
        utpop->setbwgain(0);
        pdot = max(1, trnpat / 20);
        for (int p = 0; p < trnpat; p++) {
            if (myverbosity > 2 and p % pdot == 0)
                fprintf(stderr, ".");
            //Pop::resetall();
            //inpop->setinput(trainpatfac->getfpat(p));
            //utpop->setinput(trainlblfac->getfpat(p));
            //for (int loop = 0; loop < nloop; loop++) {
            //    if (loop == nloop - 1)
            //        prns[1] = 1;
            //    Pop::updsupall();
            //    Pop::updactall();
            //    Prj::upddenactall();
            //    Prj::updtracesall(prns);
            //    Pop::resetbwsupall();
            //    Prj::updbwsupall();
            //    Prj::contributeall();
            //    prns[1] = 0;
            //    advance();
            //}
            copyDataToBuffer(inpop, hidpop, utpop, ihprj, huprj,
                             rndPoisson_in, rndPoisson_hid, rndPoisson_ut,
                             Hihjhi_ih, Chjhi_ih,
                             Zj_ih, Zi_ih, Pj_ih, Pi_ih, Pji_ih, Wji_ih, Bj_ih,
                             Zj_hu, Zi_hu, Pj_hu, Pi_hu, Pji_hu, Wji_hu, Bj_hu,
                             needsupdbw, constant_hbm, SUPERVISED);
            BCPNN_Kernel(trainpatfac->getfpat(p), trainlblfac->getfpat(p), utpop->act, SUPERVISED, rndPoisson_hid,
                         Hihjhi_ih, Chjhi_ih, needsupdbw,
                         Zj_ih, Zi_ih, Pj_ih, Pi_ih, Pji_ih, Bj_ih, Wji_ih,
                         Zj_hu, Zi_hu, Pj_hu, Pi_hu, Pji_hu, Bj_hu, Wji_hu,
                         constant_hbm);
            copyBufToData(ihprj, huprj, needsupdbw,
                          Zj_ih, Zi_ih, Pj_ih, Pi_ih, Pji_ih, Wji_ih, Bj_ih,
                          Zj_hu, Zi_hu, Pj_hu, Pi_hu, Pji_hu, Wji_hu, Bj_hu);

        }
        utpop->setbwgain(1);
        if (myverbosity > 2) {
            printf("\n");
            fflush(stdout);
        }
        huprj->updbw();
        fill(prns.begin(), prns.end(), 0);
        // Recall
        if (nrep == 1) {
            printf("Recall (on training patterns) (%d)\n", simstep);
            fflush(stdout);
        }
        int minpat = min(trnpat, tenpat);
        pdot = max(1, minpat / 20);
        ncorr = 0;
        for (int p = 0; p < minpat; p++) {
            if (myverbosity > 2 and p % pdot == 0)
                fprintf(stderr, ".");
            //Pop::resetall();
            //inpop->setinput(trainpatfac->getfpat(p));
            //utpop->setinput();
            //for (int loop = 0; loop < nloop; loop++) {
            //    Pop::updsupall();
            //    Pop::updactall();
            //    Prj::upddenactall();
            //    Prj::updtracesall();
            //    Pop::resetbwsupall();
            //    Prj::updbwsupall();
            //    Prj::contributeall();
            //    if (loop == nloop - 1)
            //        ncorr += analys1->iscorr(utpop->act, trainlblfac->getfpat(p));
            //    advance();
            //}
            copyDataToBuffer(inpop, hidpop, utpop, ihprj, huprj,
                             rndPoisson_in, rndPoisson_hid, rndPoisson_ut, Hihjhi_ih, Chjhi_ih,
                             Zj_ih, Zi_ih, Pj_ih, Pi_ih, Pji_ih, Wji_ih, Bj_ih,
                             Zj_hu, Zi_hu, Pj_hu, Pi_hu, Pji_hu, Wji_hu, Bj_hu,
                             needsupdbw, constant_hbm, INFERENCES);
            BCPNN_Kernel(trainpatfac->getfpat(p), NULL, utpop->act, INFERENCES, rndPoisson_hid,
                         Hihjhi_ih, Chjhi_ih, needsupdbw,
                         Zj_ih, Zi_ih, Pj_ih, Pi_ih, Pji_ih, Bj_ih, Wji_ih, 
                         Zj_hu, Zi_hu, Pj_hu, Pi_hu, Pji_hu, Bj_hu, Wji_hu,
                         constant_hbm);
            copyBufToData(ihprj, huprj, needsupdbw,
                          Zj_ih, Zi_ih, Pj_ih, Pi_ih, Pji_ih,Wji_ih, Bj_ih,
                          Zj_hu, Zi_hu, Pj_hu, Pi_hu, Pji_hu, Wji_hu, Bj_hu);
            ncorr += analys1->iscorr(utpop->act, trainlblfac->getfpat(p));

        }
        trfcorr = ncorr / (float)minpat;
        trfcorrs.push_back(trfcorr);
        if (myverbosity > 2)
            printf("\n");
        if (nrep == 1) {
            printf("Recall (on test patterns) (%d)\n", simstep);
            fflush(stdout);
        }
        pdot = max(1, tenpat / 20);
        ncorr = 0;
        for (int p = 0; p < tenpat; p++) {
            if (myverbosity > 2 and p % pdot == 0)
                fprintf(stderr, ".");
            //Pop::resetall();
            //inpop->setinput(testpatfac->getfpat(p));
            //utpop->setinput();
            //for (int loop = 0; loop < nloop; loop++) {
            //    Pop::updsupall();
            //    Pop::updactall();
            //    Prj::upddenactall();
            //    Prj::updtracesall();
            //    Pop::resetbwsupall();
            //    Prj::updbwsupall();
            //    Prj::contributeall();
            //    if (loop == nloop - 1)
            //        ncorr += analys1->iscorr(utpop->act, testlblfac->getfpat(p));
            //    advance();
            //}
            copyDataToBuffer(inpop, hidpop, utpop, ihprj, huprj,
                             rndPoisson_in, rndPoisson_hid, rndPoisson_ut, Hihjhi_ih, Chjhi_ih,
                             Zj_ih, Zi_ih, Pj_ih, Pi_ih, Pji_ih, Wji_ih, Bj_ih,
                             Zj_hu, Zi_hu, Pj_hu, Pi_hu, Pji_hu, Wji_hu, Bj_hu,
                             needsupdbw, constant_hbm, INFERENCES);
            BCPNN_Kernel(testpatfac->getfpat(p), NULL, utpop->act, INFERENCES, rndPoisson_hid,
                         Hihjhi_ih, Chjhi_ih, needsupdbw,
                         Zj_ih, Zi_ih, Pj_ih, Pi_ih, Pji_ih, Bj_ih, Wji_ih, 
                         Zj_hu, Zi_hu, Pj_hu, Pi_hu, Pji_hu, Bj_hu, Wji_hu,
                         constant_hbm);
            copyBufToData(ihprj, huprj, needsupdbw,
                          Zj_ih, Zi_ih, Pj_ih, Pi_ih, Pji_ih,Wji_ih, Bj_ih,
                          Zj_hu, Zi_hu, Pj_hu, Pi_hu, Pji_hu, Wji_hu, Bj_hu);


            ncorr += analys1->iscorr(utpop->act, testlblfac->getfpat(p));

        }
        tefcorr = ncorr / (float)tenpat;
        tefcorrs.push_back(tefcorr);
        if (nrep == 1 or myverbosity > 1)
            if (myverbosity > 2)
                printf("\n");
        printf("Fraction recall: %.1f %% (train) %.1f %% (test)\n",
               100 * trfcorr, 100 * tefcorr);
        fflush(stdout);
        if (seed > 0)
            seed += 17;
    }
    if (nrep > 1 and myverbosity > 0) {
        printf("Mean fraction recall:\n");
        printf("mean = %.1f std = %.2f sem = %.2f %% (train)\n",
               100 * analys1->mean(trfcorrs), 100 * analys1->std(trfcorrs), 100 * analys1->sem(trfcorrs));
        printf("mean = %.1f std = %.2f sem = %.2f %% (test)\n",
               100 * analys1->mean(tefcorrs), 100 * analys1->std(tefcorrs), 100 * analys1->sem(tefcorrs));
        fflush(stdout);
    }
    if (myverbosity > 1) {
        if (myverbosity > 3)
            printf("Max memory footprint: Pops = %.1f (kB) Prjs = %.1f (MB)\n",
                   Pop::nallocbyteall() / 1000.0, Prj::nallocbyteall() / 1000000.0);
        printf("Total simsteps = %d, time elapsed = %.3f sec\n",
               simstep, getDiffTime(total_time) / 1000);
        fflush(stdout);
    }
    return 0;
}
