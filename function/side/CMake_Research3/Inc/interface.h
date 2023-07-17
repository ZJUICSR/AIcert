#ifndef _INTERFACE_H_
#define _INTERFACE_H_

extern "C"{
    void cpa_(char* inFile, char* outFile);
    void dpa_(char* inFile, char* outFile);
    void spa_(char* inFile, char* outFile);
    void hpa_(char* inFile, char* outFile);
    void ttest_(char* inFile, char* outFile);
    void X2test_(char* inFile, char* outFile);
}


#endif