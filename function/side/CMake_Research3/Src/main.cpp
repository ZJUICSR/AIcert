#include <stdio.h>
#include<time.h>
#include "../Inc/CNNModel/cifar10_NN_example.h"
#include "../Inc/side_channel_attack_methods.h"
#include "../Inc/TRACE/Trace.h"




int main(int argc, char const *argv[])
{
  //wt = 47 : 21
  //wt = 2 : 75
  int start;
  int end;
    
  
  Parameters param;
  param.setSampleFile("trs/dpa/elmotrace-9/elmotracegaus_dpa_-9.trs");
  param.setRandFile("trs/dpa/elmotrace-9/randdata-9.trs");
  param.setOutFile("trs/dpa/elmotrace-9/dpa_out-9.txt");
  param.setAttackIndex(0);
  param.setPointNumStart(108);
  param.setPointNumEnd(112);
  param.setTraceNum(5000);

  start=clock();

  differentialPowerAnalysis_correlation_distinguish(&param, cifar10_nn_run_cpa_dpa);
  // differentialPowerAnalysis(&param, cifar10_nn_run_cpa);
  // correlationPowerAnalysis(&param, cifar10_nn_run_cpa);
  // correlationPowerAnalysis_correlation_distinguish(&param, cifar10_nn_run_cpa);

  end=clock();
  printf("time:%d\n", end-start);

  param.~Parameters();

#if 0
  Trace traceS("trs/elmotrace-9/elmotrace-9gaus_small.trs");
  Trace traceR("trs/elmotrace-9/randdata-9.trs");
  TrsData trsDataS;
  TrsData trsDataR;

  TrsData trsData;
  trsData.samples=new float[200];
  trsData.data=new uint8_t[3072];

  ofstream* outfile=new ofstream();   
  outfile->open("trs/elmotrace-9/elmotracegaus_cpa_-9.trs", ios::out | ios::binary | ios::trunc);

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
    traceS.readIndexTrace(&trsDataS,i);
    traceR.readIndexTrace(&trsDataR,i);

    memcpy(trsData.samples, trsDataS.samples,200*sizeof(float));
    memcpy(trsData.data, trsDataR.data,3072*sizeof(uint8_t)); 
    Trace::writeNext(outfile, &trsData, trsHead_result);
  }
    
  
  #endif

  system("pause");
  return 0;
}
