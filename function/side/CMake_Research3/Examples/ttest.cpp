#include <stdio.h>
#include<time.h>
#include "../Inc/interface.h"
#include  "../Inc/side_channel_attack_methods.h"
#include "../Inc/CNNModel/cifar10_NN_example.h"


void ttest(char* inFile, char* outFile){

    printf("start:\n");

    #if 1

    int start;
    int end;

    int num[SIZE] = {0};
    selectParenthesesNum(inFile, num);
        
    Parameters param;
    param.setSampleFile(inFile);
    param.setRandFile(inFile);//"./CMake_Research3/Trs/random/cpa/randdata_cpa_-9.trs"
    param.setOutFile(outFile);
    param.setAttackIndex(num[0]);
    param.setPointNumStart(1);
    param.setPointNumEnd(123);
    param.setTraceNum(5000);
    param.setWtForWhiteBoxTest(num[3]);

    start=clock();
    ttest_non_specific(&param, cifar10_nn_run_cpa_dpa);
    end=clock();

    printf("time:%d\n", end-start);

    // param.~Parameters();

    #endif

}

void ttest_(char* inFile, char* outFile){
    ttest(inFile, outFile);
}