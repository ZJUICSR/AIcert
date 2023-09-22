#include <stdio.h>
#include<time.h>
#include "../Inc/interface.h"
#include  "../Inc/side_channel_attack_methods.h"
#include "../Inc/CNNModel/cifar10_NN_example.h"

#if ISTHREAD

int spa(char* inFile, char* outFile) {
    char* infile = "./50-all-1.trs";
    char* outfile = "./Outfile/";

    Parameters param;
    param.setSampleFile(inFile);
    param.setOutFile(outFile);
    simplePowerAnalysis(&param);
    
    return 0;
}

void spa_(char* inFile, char* outFile){
    spa(inFile, outFile);
}

#else

int spa(char* inFile, char* outFile){

}

void spa_(char* inFile, char* outFile){
    
}

#endif