#include <stdio.h>
#include<time.h>
#include "../Inc/interface.h"
#include  "../Inc/side_channel_attack_methods.h"
#include "../Inc/CNNModel/cifar10_NN_example.h"




void dpa(char* inFile, char* outFile)
{
    printf("start:\n");
    //wt = 47 : 21
    //wt = 2 : 75
    int start;
    int end;

    int num[SIZE] = {0};
    selectParenthesesNum(inFile, num);

    Parameters param;
    param.setSampleFile(inFile);
    param.setRandFile(inFile);//"./CMake_Research3/Trs/random/dpa/randdata_dpa_-9.trs"
    param.setOutFile(outFile);
    param.setAttackIndex(num[0]/100, num[0]/10%10, num[0]%10);
    param.setPointNumStart(num[1]);
    param.setPointNumEnd(num[2]);
    param.setTraceNum(5000);

    start=clock();
    differentialPowerAnalysis_correlation_distinguish_optimize(&param, cifar10_nn_run_cpa_dpa);
    end=clock();

    printf("time:%d\n", end-start);

    // param.~Parameters();


}

void dpa_(char* inFile, char* outFile){
    dpa(inFile, outFile);
}

