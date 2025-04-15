#include <vector>
#include <string>
#include <sys/time.h>

#include "Globals.h"
#include "Pop.h"
#include "Prj.h"
#include "Parseparam.h"
#include "PatternFactory.h"
#include "Analys.h"
#include "BCPNN_Kernel.h"

#include "xcl2.hpp"
#include <algorithm>

template <typename T>
using AlignedVector = std::vector<T, aligned_allocator<T>>;

using namespace std;
using namespace Globals;

long seed = 0;
int Hx = 5, Mx = 5, Nx, Hin = Hx, Min = Mx, Nin, Hhid = Hin, Mhid = Min, Hut = 1, Mut = 10, Nut, nhlayer = 1, updconnint = -1;
bool inbinarize = false, doshuffle = true, logenabled = true;
float eps = -1, maxfq = 1. / timestep, taup = -1, bgain = 1, wgain = 1, again = 1, cthres = 0.1, nampl = 0,
      nfreq = 1, nswap = 0, nrepl = 0, swaprthr = 1.1, replrthr = 1, replthr = -1;
string lrule = "BCP", actfn = "WTA", patype = "hrand";
string trainpatfile = "", trainlblfile = "", testpatfile = "", testlblfile = "";
int nactHi = 0, nsilHi = 0, nrep = 1, nepoch = 1, trnpat = 10, tenpat = -1, nloop = -1, nhblank = 0, nhflip = 0, myverbosity = 1;
PatternFactory *trainpatfac = nullptr, *trainlblfac = nullptr, *testpatfac = nullptr, *testlblfac = nullptr;
Pop *inpop, *hidpop, *utpop;
Prj *ihprj, *huprj;

std::vector<float> rndPoisson_hid;
std::vector<int> Hihjhi_ih, Chjhi_ih;
std::vector<float> Zj_ih, Zi_ih, Pj_ih, Pi_ih, Pji_ih, Wji_ih, Wji_ih1, Wji_ih2, Bj_ih;
std::vector<float> Zj_hu, Zi_hu, Pj_hu, Pi_hu, Pji_hu, Wji_hu, Bj_hu;
std::vector<char> needsupdbw(2);
std::vector<float> constant_hbm(21);
float bgain_ih = 1, wgain_ih = 1, ewgain_ih = 1, iwgain_ih = 1;
float bgain_hu = 1, wgain_hu = 1, ewgain_hu = 1, iwgain_hu = 1;
float igain_pop[NumberPop] = {1, 1, 1};
float bwgain_pop[NumberPop] = {1, 1, 1};
float nampl_pop = 0;
float nfreq_pop = 1;
float taumdt_pop[NumberPop] = {1, 1, 1};
float prntaudpt[2] = {0, 0};

struct timeval total_time;

int modeOps = UNSUPERVISED;

auto check_and_copy = [](float* dest, const AlignedVector<float>& src, int size, const std::string& name) {
    for (int i = 0; i < size; i++) {
        dest[i] = src[i];
        if (std::isnan(src[i])) {
            std::cout << "NaN detected in " << name << " at index " << i << std::endl;
        }
    }
};

void allocVarKernel() {
    rndPoisson_hid.resize(N_hid);
    Hihjhi_ih.resize(H_hid * denHi_ih);
    Chjhi_ih.resize(H_hid * denHi_ih);
    Zj_ih.resize(N_hid);
    Zi_ih.resize(H_hid * denNi_ih);
    Pj_ih.resize(N_hid);
    Pi_ih.resize(H_hid * denNi_ih);
    Pji_ih.resize(N_hid * denNi_ih);
    Wji_ih.resize(N_hid * denNi_ih);
    Wji_ih1.resize(N_hid * denNi_ih);
    Wji_ih2.resize(N_hid * denNi_ih);
    Bj_ih.resize(N_hid);
    Zj_hu.resize(N_ut);
    Zi_hu.resize(H_ut * denNi_hu);
    Pj_hu.resize(N_ut);
    Pi_hu.resize(H_ut * denNi_hu);
    Pji_hu.resize(N_ut * denNi_hu);
    Wji_hu.resize(N_ut * denNi_hu);
    Bj_hu.resize(N_ut);
}

void copyVectorToData(Prj *ihprj, Prj *huprj, AlignedVector<float>& Pj_ih_cl, AlignedVector<float>& Bj_ih_cl,
                      AlignedVector<float>& Pi_ih_cl, AlignedVector<float>& Pji_ih_cl, AlignedVector<float>& Wji_ih_cl,
                      AlignedVector<float>& Pj_hu_cl, AlignedVector<float>& Bj_hu_cl,
                      AlignedVector<float>& Pi_hu_cl, AlignedVector<float>& Pji_hu_cl, AlignedVector<float>& Wji_hu_cl,
                      AlignedVector<char>& needsupdbw_cl){
    check_and_copy(ihprj->Bj, Bj_ih_cl, N_hid, "Bj_ih_cl");
    check_and_copy(ihprj->Pj, Pj_ih_cl, N_hid, "Pj_ih_cl");
    check_and_copy(ihprj->Pi, Pi_ih_cl, H_hid * denNi_ih, "Pi_ih_cl");
    check_and_copy(ihprj->Pji, Pji_ih_cl, N_hid * denNi_ih, "Pji_ih_cl");
    check_and_copy(ihprj->Wji, Wji_ih_cl, N_hid * denNi_ih, "Wji_ih_cl");

    check_and_copy(huprj->Bj, Bj_hu_cl, N_ut, "Bj_hu_cl");
    check_and_copy(huprj->Pj, Pj_hu_cl, N_ut, "Pj_hu_cl");
    check_and_copy(huprj->Pi, Pi_hu_cl, H_ut * denNi_hu, "Pi_hu_cl");
    check_and_copy(huprj->Pji, Pji_hu_cl, N_ut * denNi_hu, "Pji_hu_cl");
    check_and_copy(huprj->Wji, Wji_hu_cl, N_ut * denNi_hu, "Wji_hu_cl");

    ihprj->needsupdbw = needsupdbw_cl[0];
    huprj->needsupdbw = needsupdbw_cl[1];
}

void copyDataToVector(Prj *ihprj, Prj *huprj, AlignedVector<float>& Pj_ih_cl, AlignedVector<float>& Bj_ih_cl,
                      AlignedVector<float>& Pi_ih_cl, AlignedVector<float>& Pji_ih_cl, AlignedVector<float>& Wji_ih_cl,
                      AlignedVector<float>& Wji_ih1_cl, AlignedVector<float>& Wji_ih2_cl,
                      AlignedVector<float>& Pj_hu_cl, AlignedVector<float>& Bj_hu_cl,
                      AlignedVector<float>& Pi_hu_cl, AlignedVector<float>& Pji_hu_cl, AlignedVector<float>& Wji_hu_cl,
                      AlignedVector<char>& needsupdbw_cl, AlignedVector<float>& rndPoisson_hid_cl,
                      AlignedVector<int>& Hihjhi_ih_cl, AlignedVector<int>& Chjhi_ih_cl,
                      AlignedVector<float>& constant_hbm_cl, int modeOps) {

    // Helper to check and assign float values
    auto assign_with_nan_check = [](float& dest, float value, const std::string& name, int idx) {
        dest = value;
        if (std::isnan(value)) {
            std::cout << "NaN detected in " << name << " at index " << idx << std::endl;
        }
    };

    for (int i = 0; i < N_ut; ++i) {
        assign_with_nan_check(Bj_hu_cl[i], huprj->Bj[i], "Bj_hu", i);
        assign_with_nan_check(Pj_hu_cl[i], huprj->Pj[i], "Pj_hu", i);
    }

    for (int i = 0; i < N_hid; ++i) {
        rndPoisson_hid_cl[i] = gnextpoisson();
        assign_with_nan_check(Bj_ih_cl[i], ihprj->Bj[i], "Bj_ih", i);
        assign_with_nan_check(Pj_ih_cl[i], ihprj->Pj[i], "Pj_ih", i);
    }

    for (int i = 0; i < H_hid * denHi_ih; ++i) {
        Hihjhi_ih_cl[i] = ihprj->Hihjhi[i];
        Chjhi_ih_cl[i] = ihprj->Chjhi[i];
    }

    for (int i = 0; i < H_hid * denNi_ih; ++i) {
        assign_with_nan_check(Pi_ih_cl[i], ihprj->Pi[i], "Pi_ih", i);
    }

    for (int i = 0; i < N_hid * denNi_ih; ++i) {
        float wji = ihprj->Wji[i];
        Pji_ih_cl[i] = ihprj->Pji[i];
        Wji_ih_cl[i] = wji;
        Wji_ih1_cl[i] = wji;
        Wji_ih2_cl[i] = wji;
    }

    for (int i = 0; i < H_ut * denNi_hu; ++i) {
        assign_with_nan_check(Pi_hu_cl[i], huprj->Pi[i], "Pi_hu", i);
    }

    for (int i = 0; i < N_ut * denNi_hu; ++i) {
        Pji_hu_cl[i] = huprj->Pji[i];
        Wji_hu_cl[i] = huprj->Wji[i];
    }

    // Set constants (shared values grouped)
    constant_hbm_cl = {
        ihprj->bgain, ihprj->wgain, ihprj->ewgain, ihprj->iwgain,
        huprj->bgain, huprj->wgain, huprj->ewgain, huprj->iwgain,
        hidpop->nampl, hidpop->nfreq
    };

    switch (modeOps) {
        case UNSUPERVISED:
            constant_hbm_cl.insert(constant_hbm_cl.end(), { ihprj->taupdt, 0.0f });
            break;
        case SUPERVISED:
            constant_hbm_cl.insert(constant_hbm_cl.end(), { 0.0f, huprj->taupdt });
            break;
        case INFERENCES:
            constant_hbm_cl.insert(constant_hbm_cl.end(), { 0.0f, 0.0f });
            break;
    }

    constant_hbm_cl.insert(constant_hbm_cl.end(), {
        inpop->igain, hidpop->igain, utpop->igain,
        inpop->bwgain, hidpop->bwgain, utpop->bwgain,
        inpop->taumdt, hidpop->taumdt, utpop->taumdt
    });

    needsupdbw_cl = { ihprj->needsupdbw, huprj->needsupdbw };
}

void copyBufToData(Prj *ihprj, Prj *huprj) {
    // Lambda to check for NaN values in a std::vector
    auto hasNaN = [](const std::vector<float>& arr, const char *name) -> bool {
        for (size_t i = 0; i < arr.size(); i++) {
            if (std::isnan(arr[i])) {
                printf("Error: NaN detected in %s at index %zu\n", name, i);
                return true;
            }
        }
        return false;
    };

    // Check all arrays before copying
    bool nanFound = false;
    nanFound |= hasNaN(Pj_ih, "Pj_ih");
    nanFound |= hasNaN(Pi_ih, "Pi_ih");
    nanFound |= hasNaN(Pji_ih, "Pji_ih");
    nanFound |= hasNaN(Wji_ih, "Wji_ih");
    nanFound |= hasNaN(Bj_ih, "Bj_ih");
    nanFound |= hasNaN(Pj_hu, "Pj_hu");
    nanFound |= hasNaN(Pi_hu, "Pi_hu");
    nanFound |= hasNaN(Pji_hu, "Pji_hu");
    nanFound |= hasNaN(Wji_hu, "Wji_hu");
    nanFound |= hasNaN(Bj_hu, "Bj_hu");

    if (nanFound) {
        printf("Error: NaN values detected. Copy aborted.\n");
        return;
    }

    std::copy(Pj_ih.begin(), Pj_ih.end(), ihprj->Pj);
    std::copy(Pi_ih.begin(), Pi_ih.end(), ihprj->Pi);
    for (size_t i = 0; i < Pji_ih.size(); i++) {
        ihprj->Pji[i] = Pji_ih[i];
        ihprj->Wji[i] = Wji_ih[i];
    }
    std::copy(Bj_ih.begin(), Bj_ih.end(), ihprj->Bj);

    std::copy(Pj_hu.begin(), Pj_hu.end(), huprj->Pj);
    std::copy(Pi_hu.begin(), Pi_hu.end(), huprj->Pi);
    std::copy(Pji_hu.begin(), Pji_hu.end(), huprj->Pji);
    std::copy(Wji_hu.begin(), Wji_hu.end(), huprj->Wji);
    std::copy(Bj_hu.begin(), Bj_hu.end(), huprj->Bj);

    ihprj->needsupdbw = needsupdbw[0];
    huprj->needsupdbw = needsupdbw[1];
}


void copyDataToBuffer(Pop *inpop, Pop *hidpop, Pop *utpop, Prj *ihprj, Prj *huprj, int modeOps) {
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
    constant_hbm[10] = (modeOps == UNSUPERVISED) ? ihprj->taupdt : 0.0f;
    constant_hbm[11] = (modeOps == SUPERVISED) ? huprj->taupdt : 0.0f;
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

    gsetpoissonmean(hidpop->nfreq);
    for (int i = 0; i < N_hid; i++)
        rndPoisson_hid[i] = gnextpoisson();

    std::copy(ihprj->Hihjhi, ihprj->Hihjhi + H_hid * denHi_ih, Hihjhi_ih.begin());
    std::copy(ihprj->Chjhi, ihprj->Chjhi + H_hid * denHi_ih, Chjhi_ih.begin());
    std::copy(ihprj->Pj, ihprj->Pj + N_hid, Pj_ih.begin());
    std::copy(ihprj->Pi, ihprj->Pi + H_hid * denNi_ih, Pi_ih.begin());
    std::copy(ihprj->Bj, ihprj->Bj + N_hid, Bj_ih.begin());

    for (int i = 0; i < N_hid * denNi_ih; i++) {
        Pji_ih[i] = ihprj->Pji[i];
        Wji_ih[i] = ihprj->Wji[i];
        Wji_ih1[i] = ihprj->Wji[i];
        Wji_ih2[i] = ihprj->Wji[i];
    }

    std::copy(huprj->Pj, huprj->Pj + N_ut, Pj_hu.begin());
    std::copy(huprj->Pi, huprj->Pi + H_ut * denNi_hu, Pi_hu.begin());
    std::copy(huprj->Pji, huprj->Pji + N_ut * denNi_hu, Pji_hu.begin());
    std::copy(huprj->Wji, huprj->Wji + N_ut * denNi_hu, Wji_hu.begin());
    std::copy(huprj->Bj, huprj->Bj + N_ut, Bj_hu.begin());
}

void saveVectorsToFile(const std::string &filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        fprintf(stderr, "Failed to open file for saving: %s\n", filename.c_str());
        return;
    }

    auto writeVec = [&out](auto &vec) {
        size_t size = vec.size();
        out.write(reinterpret_cast<char*>(&size), sizeof(size));
        out.write(reinterpret_cast<char*>(vec.data()), size * sizeof(typename std::remove_reference<decltype(vec)>::type::value_type));
    };

    writeVec(Hihjhi_ih);
    writeVec(Chjhi_ih);
    writeVec(needsupdbw);
    writeVec(Pj_ih);
    writeVec(Pi_ih);
    writeVec(Pji_ih);
    writeVec(Bj_ih);
    writeVec(Wji_ih);
    writeVec(Wji_ih1);
    writeVec(Wji_ih2);
    writeVec(Pj_hu);
    writeVec(Pi_hu);
    writeVec(Pji_hu);
    writeVec(Bj_hu);
    writeVec(Wji_hu);
    out.close();
    printf("Saved vectors to file: %s\n", filename.c_str());
}

void loadVectorsFromFile(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        fprintf(stderr, "Failed to open file for loading: %s\n", filename.c_str());
        return;
    }

    auto readVec = [&in](auto &vec) {
        size_t size = 0;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        vec.resize(size);
        in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(typename std::remove_reference<decltype(vec)>::type::value_type));
    };

    readVec(Hihjhi_ih);
    readVec(Chjhi_ih);
    readVec(needsupdbw);
    readVec(Pj_ih);
    readVec(Pi_ih);
    readVec(Pji_ih);
    readVec(Bj_ih);
    readVec(Wji_ih);
    readVec(Wji_ih1);
    readVec(Wji_ih2);
    readVec(Pj_hu);
    readVec(Pi_hu);
    readVec(Pji_hu);
    readVec(Bj_hu);
    readVec(Wji_hu);
    in.close();
    printf("Loaded vectors from file: %s\n", filename.c_str());
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

// Helper function to set all kernel arguments
void setKernelArguments(
    cl::Kernel &kernel,
    cl::CommandQueue &qq,
    int modeOps,
    cl::Buffer &buf_inputdata,
    cl::Buffer &buf_labeldata,
    cl::Buffer &buf_outputdata,
    cl::Buffer &buf_rndPoisson_hid,
    cl::Buffer &buf_Hihjhi_ih,
    cl::Buffer &buf_Chjhi_ih,
    cl::Buffer &buf_needsupdbw,
    cl::Buffer &buf_Pj_ih,
    cl::Buffer &buf_Pi_ih,
    cl::Buffer &buf_Pji_ih,
    cl::Buffer &buf_Bj_ih,
    cl::Buffer &buf_Wji_ih,
    cl::Buffer &buf_Wji_ih1,
    cl::Buffer &buf_Wji_ih2,
    cl::Buffer &buf_Pj_hu,
    cl::Buffer &buf_Pi_hu,
    cl::Buffer &buf_Pji_hu,
    cl::Buffer &buf_Bj_hu,
    cl::Buffer &buf_Wji_hu,
    cl::Buffer &buf_constant_hbm)
{
    cl_int err;
    int narg = 0;
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_inputdata));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_labeldata));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_outputdata));
    OCL_CHECK(err, err = kernel.setArg(narg++, modeOps));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_rndPoisson_hid));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Hihjhi_ih));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Chjhi_ih));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_needsupdbw));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Pj_ih));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Pi_ih));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Pji_ih));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Bj_ih));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Wji_ih));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Wji_ih1));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Wji_ih2));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Pj_hu));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Pi_hu));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Pji_hu));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Bj_hu));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_Wji_hu));
    OCL_CHECK(err, err = kernel.setArg(narg++, buf_constant_hbm));
    OCL_CHECK(err, err = qq.enqueueMigrateMemObjects({buf_inputdata, buf_labeldata, buf_rndPoisson_hid, 
        buf_Hihjhi_ih, buf_Chjhi_ih, buf_Pj_ih, buf_Pi_ih, buf_Pji_ih, buf_Bj_ih, 
        buf_Wji_ih, buf_Wji_ih1, buf_Wji_ih2, buf_Pj_hu, buf_Pi_hu, buf_Pji_hu, buf_Bj_hu, buf_Wji_hu}, 0));

    OCL_CHECK(err, err = qq.enqueueTask(kernel));

    OCL_CHECK(err, err = qq.enqueueMigrateMemObjects({buf_Pj_ih, buf_Pi_ih, buf_Pji_ih, buf_Bj_ih, 
        buf_Wji_ih, buf_Pj_hu, buf_Pi_hu, buf_Pji_hu, buf_Bj_hu, buf_Wji_hu, 
        buf_outputdata, buf_needsupdbw}, CL_MIGRATE_MEM_OBJECT_HOST));
    qq.finish();

}

int main(int argc, char **args) {
    gettimeofday(&total_time, 0);
    string paramfile = "mnistmain.par";
    std::string binaryFile;
    std::string trained_data_file = "trained_data.bin";
    if (argc > 1)
        paramfile = args[1];
    if (argc > 2)
        binaryFile = args[2];
    if (argc > 3)
        trained_data_file = args[3];
    parseparams(paramfile);

    cl_int err;
    cl::Context context;
    cl::Kernel krnl_bcpnn;
    cl::CommandQueue qq;

    std::vector<float, aligned_allocator<float>> constant_hbm_cl(21);
    std::vector<char, aligned_allocator<char>>  needsupdbw_cl(2);
    std::vector<float, aligned_allocator<float>> rndPoisson_hid_cl(N_hid);
    std::vector<int, aligned_allocator<int>>   Hihjhi_ih_cl(H_hid * denHi_ih);
    std::vector<int, aligned_allocator<int>>   Chjhi_ih_cl(H_hid * denHi_ih);
    std::vector<float, aligned_allocator<float>> Pj_ih_cl(N_hid);
    std::vector<float, aligned_allocator<float>> Bj_ih_cl(N_hid);
    std::vector<float, aligned_allocator<float>> Pi_ih_cl(H_hid * denNi_ih);
    std::vector<float, aligned_allocator<float>> Pji_ih_cl(N_hid * denNi_ih);
    std::vector<float, aligned_allocator<float>> Wji_ih_cl(N_hid * denNi_ih);
    std::vector<float, aligned_allocator<float>> Wji_ih1_cl(N_hid * denNi_ih);
    std::vector<float, aligned_allocator<float>> Wji_ih2_cl(N_hid * denNi_ih);
    std::vector<float, aligned_allocator<float>> Pj_hu_cl(N_ut);
    std::vector<float, aligned_allocator<float>> Bj_hu_cl(N_ut);
    std::vector<float, aligned_allocator<float>> Pi_hu_cl(H_ut * denNi_hu);
    std::vector<float, aligned_allocator<float>> Pji_hu_cl(N_ut * denNi_hu);
    std::vector<float, aligned_allocator<float>> Wji_hu_cl(N_ut * denNi_hu);
    std::vector<float, aligned_allocator<float>> inputdata(N_in);
    std::vector<float, aligned_allocator<float>> labeldata(N_ut);
    std::vector<float, aligned_allocator<float>> outputdata(N_ut);

    allocVarKernel();

    auto devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, qq = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_bcpnn = cl::Kernel(program, "BCPNN_Kernel", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // Create device buffers using CL_MEM_ALLOC_HOST_PTR
    OCL_CHECK(err, cl::Buffer buf_rndPoisson_hid(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*N_hid, rndPoisson_hid_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Hihjhi_ih(context,      CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int)*H_hid*denHi_ih, Hihjhi_ih_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Chjhi_ih(context,       CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int)*H_hid*denHi_ih, Chjhi_ih_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Pj_ih(context,          CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*N_hid, Pj_ih_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Pi_ih(context,          CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*H_hid*denNi_ih, Pi_ih_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Pji_ih(context,         CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*N_hid*denNi_ih, Pji_ih_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Wji_ih(context,         CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*N_hid*denNi_ih, Wji_ih_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Wji_ih1(context,        CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*N_hid*denNi_ih, Wji_ih1_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Wji_ih2(context,        CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*N_hid*denNi_ih, Wji_ih2_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Bj_ih(context,          CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*N_hid, Bj_ih_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Pj_hu(context,          CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*N_ut, Pj_hu_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Pi_hu(context,          CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*H_ut*denNi_hu, Pi_hu_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Pji_hu(context,         CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*N_ut*denNi_hu, Pji_hu_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Wji_hu(context,         CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*N_ut*denNi_hu, Wji_hu_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_Bj_hu(context,          CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*N_ut, Bj_hu_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_needsupdbw(context,     CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(char)*2, needsupdbw_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_constant_hbm(context,   CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*21, constant_hbm_cl.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_inputdata(context,      CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(float)*N_in, inputdata.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_labeldata(context,      CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(float)*N_ut, labeldata.data(), &err));
    OCL_CHECK(err, cl::Buffer buf_outputdata(context,     CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(float)*N_ut, outputdata.data(), &err));
    
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

        copyDataToBuffer(inpop, hidpop, utpop, ihprj, huprj, INFERENCES);
        loadVectorsFromFile(trained_data_file);
        copyBufToData(ihprj, huprj);
        modeOps = INFERENCES;
        copyDataToVector(ihprj, huprj, Pj_ih_cl, Bj_ih_cl, Pi_ih_cl, Pji_ih_cl, Wji_ih_cl, Wji_ih1_cl, Wji_ih2_cl,
                         Pj_hu_cl, Bj_hu_cl, Pi_hu_cl, Pji_hu_cl, Wji_hu_cl, needsupdbw_cl, rndPoisson_hid_cl,
                         Hihjhi_ih_cl, Chjhi_ih_cl, constant_hbm_cl, modeOps);

        // Recall
        if (nrep == 1) {
            printf("Recall (on training patterns) (%d)\n", simstep);
            fflush(stdout);
        }
        int minpat = min(trnpat, tenpat);
        int pdot = max(1, minpat / 20);
        ncorr = 0;
        for (int p = 0; p < minpat; p++) {
            if (myverbosity > 2 and p % pdot == 0)
                fprintf(stderr, ".");
             for(int i = 0; i < N_in; i++) {
                 inputdata[i] = trainpatfac->getfpat(p)[i];
             }
             for (int i = 0; i < N_hid; ++i) {
                rndPoisson_hid_cl[i] = gnextpoisson();
            }
             setKernelArguments(
                 krnl_bcpnn, qq, modeOps, buf_inputdata, buf_labeldata, buf_outputdata, buf_rndPoisson_hid,
                 buf_Hihjhi_ih, buf_Chjhi_ih, buf_needsupdbw, buf_Pj_ih, buf_Pi_ih, buf_Pji_ih, buf_Bj_ih,
                 buf_Wji_ih, buf_Wji_ih1, buf_Wji_ih2, buf_Pj_hu, buf_Pi_hu, buf_Pji_hu, buf_Bj_hu, buf_Wji_hu,
                 buf_constant_hbm);
 
             for(int i = 0; i < N_ut; i++) {
                 utpop->act[i] = outputdata[i];
             }
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
             for(int i = 0; i < N_in; i++) {
                 inputdata[i] = testpatfac->getfpat(p)[i];
             }
             for (int i = 0; i < N_hid; ++i) {
                rndPoisson_hid_cl[i] = gnextpoisson();
            }
             setKernelArguments(
                 krnl_bcpnn, qq, modeOps, buf_inputdata, buf_labeldata, buf_outputdata, buf_rndPoisson_hid,
                 buf_Hihjhi_ih, buf_Chjhi_ih, buf_needsupdbw, buf_Pj_ih, buf_Pi_ih, buf_Pji_ih, buf_Bj_ih,
                 buf_Wji_ih, buf_Wji_ih1, buf_Wji_ih2, buf_Pj_hu, buf_Pi_hu, buf_Pji_hu, buf_Bj_hu, buf_Wji_hu,
                 buf_constant_hbm);
 
             for(int i = 0; i < N_ut; i++) {
                 utpop->act[i] = outputdata[i];
             }
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
    // -------------------------------------------------
    if (inpop) {
        delete inpop;
        inpop = nullptr;
    }
    if (hidpop) {
        delete hidpop;
        hidpop = nullptr;
    }
    if (utpop) {
        delete utpop;
        utpop = nullptr;
    }
    if (ihprj) {
        delete ihprj;
        ihprj = nullptr;
    }
    if (huprj) {
        delete huprj;
        huprj = nullptr;
    }

    // Delete dynamically allocated arrays
    rndPoisson_hid.clear();
    Hihjhi_ih.clear();
    Chjhi_ih.clear();
    Zj_ih.clear();
    Zi_ih.clear();
    Pj_ih.clear();
    Pi_ih.clear();
    Pji_ih.clear();
    Wji_ih.clear();
    Wji_ih1.clear();
    Wji_ih2.clear();
    Bj_ih.clear();
    Zj_hu.clear();
    Zi_hu.clear();
    Pj_hu.clear();
    Pi_hu.clear();
    Pji_hu.clear();
    Wji_hu.clear();
    Bj_hu.clear();
    needsupdbw.clear();
    constant_hbm.clear();

    // Optionally, force OpenCL objects to be cleaned up explicitly.
    // For example, reset the command queue, context, and kernel objects:
    krnl_bcpnn = cl::Kernel();   // Reset kernel
    qq = cl::CommandQueue();       // Reset command queue
    context = cl::Context();       // Reset context

    std::exit(0);
}
