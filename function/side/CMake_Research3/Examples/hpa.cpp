#include <stdio.h>
#include<time.h>
#include "../Inc/interface.h"
#include  "../Inc/side_channel_attack_methods.h"
#include "../Inc/CNNModel/cifar10_NN_example.h"



void hpa(char* inFile, char* outFile){

    #if 1

    int start;
    int end;

    int num[SIZE] = {0};
    selectParenthesesNum(inFile, num);
    printf("%d %d %d\n", num[0], num[1], num[2]);

    printf("start:\n");
    Parameters param;
    param.setSampleFile(inFile);
    param.setRandFile(inFile);//"./CMake_Research3/Trs/random/cpa/randdata_cpa_-9.trs"
    param.setOutFile(outFile);
    param.setAttackIndex(num[0]/100, num[0]/10%10, num[0]%10);
    param.setPointNumStart(num[1]);
    param.setPointNumEnd(num[2]);
    param.setTraceNum(9);
    param.setMidvaluePerTrace(784);

    start=clock();
    horizontalPowerAnalysis_correlation_distinguish(&param, cifar10_nn_run_hpa);
    end=clock();

    printf("time:%d\n", end-start);

    // param.~Parameters();

    #endif

}

void hpa_(char* inFile, char* outFile){
    hpa(inFile, outFile);
}