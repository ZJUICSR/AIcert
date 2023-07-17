#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include "../../Inc/side_channel_attack_methods.h"
#include "../../Inc/TRACE/Trace.h"
#include "../../Inc/BASETOOL/BaseTools.h"
#include "../../Inc/CNNModel/cnn.h"

void ttest_non_specific(Parameters* param, int8_t* (*f)(Parameters*)){
    
    //fmap初始化
    q7_t fmap[param->getFmapNum()]={0};
    fmapInitial(param, fmap);

    //一次读量初始化
	float* sample[param->getPointNum()] = {nullptr};
    for(int i=0;i< param->getPointNum() ;i++){
        sample[i]=(float*)malloc(param->getTraceNum()*sizeof(float));
        std::memset(sample[i], 0.0, param->getTraceNum() * sizeof(float));
    }

    //预读量
    float* sampler[param->getTraceNum()] = {nullptr};
    for(int i=0;i< param->getTraceNum() ;i++){
        sampler[i]=(float*)malloc(param->getPointNum()*sizeof(float));
        std::memset(sampler[i], 0.0, param->getPointNum() * sizeof(float));
    }printf("done1");

    Trace traceR(param->getRandFile());
	Trace traceS(param->getSampleFile());
    TrsData trsDataR;
	TrsData trsDataS;

    TrsData trsData_result;
    trsData_result.samples = (float*)malloc(param->getPointNum() * sizeof(float));
    std::memset(trsData_result.samples, 0.0, param->getPointNum() * sizeof(float));
    uint8_t sort[param->getTraceNum()] = {0};

    //attackindex -->> i j k m n l
    int attackIndex = param->getAttackIndex();
    param->setForI(attackIndex/75);//
    param->setForJ(2);
    param->setForK(2);
    param->setForM(attackIndex%75/15);//
    param->setForN(attackIndex%75%15/3);//
    param->setForL(attackIndex%3);//

    

    int counts = 0;
    for(int i = 0; i < param->getTraceNum(); i++){

        //读一条
        traceS.readIndexTrace(&trsDataS,i);
        traceR.readIndexTrace(&trsDataR,i); 

        param->setImageDataPoint(trsDataR.data);//param读imagedata参数
        fmap[param->getAttackIndex()] = param->getWtForWhiteBoxTest();

        (*f)(param);
        // hw[i] = param->getMidHW();

        if(param->getMidHW()>25){
            counts++;
            sort[i] = 1;
        }

        std::memcpy(sampler[i], trsDataS.samples, param->getPointNum() * sizeof(float));

    }
    int m = 0;
    int n = counts;
    for(int i = 0; i < param->getTraceNum(); i++){
        if(sort[i]){
            for(int j=0; j < param->getPointNum() ; j++){
                sample[j][m] = sampler[i][j+param->getPointNumStart()];
            }
            m++;  
        }else{
            for(int j=0; j < param->getPointNum() ; j++){
                sample[j][n] = sampler[i][j+param->getPointNumStart()];
            }  
            n++;
        }
    }

    for(int i=0; i < param->getPointNum() ; i++){
        trsData_result.samples[i] = BaseTools::ttest(sample[i], sample[i]+counts, counts, param->getTraceNum()-counts);
    }

    ofstream* outfile=new ofstream();   
    outfile->open(param->getOutFile(), ios::out | ios::binary | ios::trunc);

    #if 1 //save as trs
    TrsHead trsHead_result;
    trsHead_result.NT = 1;
    trsHead_result.NS = param->getPointNum();
    trsHead_result.DS = 0;
    trsHead_result.YS = 1;
    trsHead_result.SC = 0x14;//float存储类型
    trsHead_result.GT_length = 0;
    trsHead_result.DC_length = 0;
    trsHead_result.XL_length = 0;
    trsHead_result.YL_length = 0;
    trsHead_result.TS = 0;
    Trace::writeHead(outfile, trsHead_result);
    Trace::writeNext(outfile, &trsData_result, trsHead_result);

    #endif

    #if 0 //save as txt
    for(int i = 0; i < param->getPointNum(); i++){
        *outfile<<trsData_result.samples[i]<<" ";
    }
    *outfile<<"\n";
    #endif

    for(int i=0;i < param->getPointNum();i++){
        free(sample[i]);
    }

    for(int i=0;i < param->getTraceNum();i++){
        free(sampler[i]);
    }

    free(trsData_result.samples);

    outfile->close();
    free(outfile);

}

void ttest_specific(){

}