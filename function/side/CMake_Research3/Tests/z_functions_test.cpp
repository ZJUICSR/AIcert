#include<stdio.h>
#include <cstring>
#include "../Inc/TRACE/Trace.h"

void printTrs(char* trs){
    Trace trace(trs);
    TrsData trsData;
    trace.readIndexTrace(&trsData, 0);
    for(int i = 0; i < 200; i++){
        printf("%f ", trsData.samples[i]);
    }
    free(trsData.samples);
    free(trsData.data);
    free(trsData.TSData);
}

void repairTrs(char* trsSrc, char* trsDis){

    Trace trace(trsSrc);
    TrsData trsDataS;
    TrsData trsData;
    trsData.samples=new float[200];
    trsData.data=new uint8_t[3072];

    ofstream* outfile=new ofstream();   
    outfile->open(trsDis, ios::out | ios::binary | ios::trunc);

    TrsHead trsHead_result;
    trsHead_result.NT = 5000;
    trsHead_result.NS = 200;
    trsHead_result.DS = 3072;
    trsHead_result.YS = 1;
    trsHead_result.SC = 0x14;//float存储类型
    trsHead_result.GT_length = 0;
    trsHead_result.DC_length = 0;
    trsHead_result.XL_length = 0;
    trsHead_result.YL_length = 0;
    trsHead_result.TS = 0;
    Trace::writeHead(outfile, trsHead_result);

    for(int i =0 ;i<5000;i++){
        trace.readIndexTrace(&trsDataS,i);
        

        memcpy(trsData.samples, trsDataS.samples,200*sizeof(float));
        memcpy(trsData.data, trsDataS.data,3072*sizeof(uint8_t)); 
        Trace::writeNext(outfile, &trsData, trsHead_result);
    }

    free(trsData.samples);
    free(trsData.data);

    outfile->close();
    free(outfile);

}

int main(int argc, char const *argv[])
{
    #if 0//打印trace

    printTrs("Trs/samples/cpa/elmotrace2/elmotracegaus_cpa_2.trs");

    #endif

    #if 0//修复trace

    char* trsSrc = "Trs/samples/dpa/elmotrace-9/elmotracegaus_dpa_-9.trs";
    char* trsDis = "Trs/samples/dpa/elmotrace-9/elmotracegaus_dpa_-9_.trs";

    repairTrs(trsSrc, trsDis);
    
    #endif

    return 0;
}
