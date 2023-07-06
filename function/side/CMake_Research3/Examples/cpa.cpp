#include <stdio.h>
#include<time.h>
#include "../Inc/interface.h"
#include  "../Inc/side_channel_attack_methods.h"
#include "../Inc/CNNModel/cifar10_NN_example.h"

void cpa(char* inFile, char* outFile){

    int start;
    int end;
        
    
    Parameters param;
    param.setSampleFile(inFile);
    // param.setRandFile("./dataset/Trs/random/randdata_9.trs");
    param.setRandFile(inFile);
    param.setOutFile(outFile);
    param.setAttackIndex(0);
    param.setPointNumStart(108);
    param.setPointNumEnd(112);
    param.setTraceNum(5000);
    // param.setTraceNum(200);

    start=clock();
    correlationPowerAnalysis_correlation_distinguish(&param, cifar10_nn_run_cpa);
    end=clock();

    printf("time:%d\n", end-start);

    // param.~Parameters();

}

void cpa_(char* inFile, char* outFile){
    cpa(inFile,outFile);
}