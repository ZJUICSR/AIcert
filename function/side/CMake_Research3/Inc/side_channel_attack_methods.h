#ifndef _SIDE_CHANNEL_ATTACKS_METHODS_H_
#define _SIDE_CHANNEL_ATTACKS_METHODS_H_
#include<stdint.h>
#include<stdlib.h>
#include <cstring>
#include "../Inc/CNNModel/arm_nnexamples_cifar10_weights.h"


#define MIN_VALUE 1e-12
#define IS_DOUBLE_ZERO(value)  (std::abs(value) < MIN_VALUE)

#define HORMID 784//hor
#define HOR_TRS 9

#define SIZE 1024

typedef struct{

    uint8_t* image_data;
    int8_t* fmap;

}FunctionParameters;

typedef struct{
    
    char samplesFile[SIZE];
    char randFile[SIZE];
    char outFile[SIZE];
    int trace_num;
    int attackindex;
    int point_num_start;
    int point_num_end;
    int mid;
    
    int guess_size;
    int fmap_num;
    int point_num;

}InParameters;

typedef struct{
    
    int forI;
    int forJ;
    int forK;
    int forM;
    int forN;
    int forL;
    int mid;
}MidParameters;

class Parameters{

    private:

    FunctionParameters* fParam;
    InParameters* in;
    MidParameters* mid;

    public:

    Parameters(){
        fParam = (FunctionParameters*)malloc(sizeof(FunctionParameters));
        in = (InParameters*)malloc(sizeof(InParameters));
        mid = (MidParameters*)malloc(sizeof(MidParameters));

        emptyFunctionParameters();
        emptyInParameters();
        emptyMidParameters();
    };

    //set and get function for InParameters* in
    //*****************************************************************
    void setSampleFile(char* fileName){
        strcpy(in->samplesFile, fileName);
    };

    void setRandFile(char* fileName){
        strcpy(in->randFile, fileName);
    };

    void setOutFile(char* fileName){
        strcpy(in->outFile, fileName);
    };

    void setTraceNum(int i){
        in->trace_num = i;
    };

    void setAttackIndex(int i){
        in->attackindex = i;
    };

    void setPointNumStart(int i){
        in->point_num_start = i;
        in->point_num = in->point_num_end - in->point_num_start + 1;
    };

    void setPointNumEnd(int i){
        in->point_num_end = i;
        in->point_num = in->point_num_end - in->point_num_start + 1;
    };

    int getTraceNum(){
        return in->trace_num;
    };

    int getAttackIndex(){
        return in->attackindex;
    };

    int getFmapNum(){
        return in->fmap_num;
    }

    int getGuessSize(){
        return in->guess_size;
    }

    int getPointNum(){
        return in->point_num;
    }

    char* getSampleFile(){
        return in->samplesFile;
    };

    char* getRandFile(){
        return in->randFile;
    };

    char* getOutFile(){
        return in->outFile;
    };

    int getPointNumStart(){
        return in->point_num_start;
    };

    int getPointNumEnd(){
        return in->point_num_end;
    };

    //****END*********************************************************************



    //set and get function for FunctionParameters* fParam
    //*******************************************************************************
    void setImageDataPoint(uint8_t* image_data){
        fParam->image_data = image_data;
    };

    void setFmapPoint(int8_t* fmap){
        fParam->fmap = fmap;
    };

    uint8_t* getImageDataPoint(){
        return fParam->image_data;
    };

    int8_t* getFmapPoint(){
        return fParam->fmap;
    };

    FunctionParameters* getFunctionParameters(){
        return fParam;
    };
    //***END*****************************************************************


    //set and get function for MidParameters* mid
    //***********************************************************************
    void setForI(int i){
        mid->forI = i;

    };

    int getForI(){
        return mid->forI;

    };

    void setForJ(int i){
        mid->forJ = i;

    };

    int getForJ(){
        return mid->forJ;

    };

    void setForK(int i){
        mid->forK = i;

    };

    int getForK(){
        return mid->forK;

    };

    void setForM(int i){
        mid->forM = i;

    };

    int getForM(){
        return mid->forM;

    };

    void setForN(int i){
        mid->forN = i;

    };

    int getForN(){
        return mid->forN;

    };

    void setForL(int i){
        mid->forL = i;

    };

    int getForL(){
        return mid->forL;

    };

    void setMid(int i){
        in->mid = i;
    };
    
    int getMid(){
        return in->mid;
    };

    MidParameters* getMidParameters(){
        return mid;
    }

    
    //********END***********************************************************

    //empty function for in/fPmara/mid
    //**********************************************************************
    void emptyInParameters(){

        strcpy(in->samplesFile, "\0");
        strcpy(in->randFile, "\0");
        strcpy(in->outFile, "\0");

        in->trace_num = 0;
        in->attackindex = 0;
        in->point_num_start = -1;
        in->point_num_end = 0;
        in->mid = 0;

        in->guess_size = 256;
        in->fmap_num = 2400;
        in->point_num = in->point_num_end - in->point_num_start + 1;

    };

    void emptyFunctionParameters(){

        fParam->image_data = NULL;
        fParam->fmap = NULL;
        
    };

    void emptyMidParameters(){
        mid->forI = 0;
        mid->forJ = 0;
        mid->forK = 0;
        mid->forM = 0;
        mid->forN = 0;
        mid->forL = 0;
        mid->mid = 0;
    };
    //********END*********************************************************


    ~Parameters(){
        // free(in);
        // free(fParam);
        // free(mid);
        
        if(in != NULL) free(in);
        if(fParam != NULL) free(fParam);
        if(mid != NULL) free(mid);
    };

};

inline void fmapInitial(Parameters* param, int8_t* fm){

    int8_t fmap[]=CONV1_WT;
    memset(fmap+param->getAttackIndex(), 0, (param->getFmapNum() - param->getAttackIndex()) * sizeof(int8_t));
    memcpy(fm, fmap, param->getAttackIndex() * sizeof(int8_t));
    param->setFmapPoint(fm);//param读fmap参数
    // for(int i =0;i<2400;i++){
    //     printf("%d ", fmap[i]);
    // }

}

//CPA
void correlationPowerAnalysis(Parameters* param, int8_t* (*f)(Parameters*));
void correlationPowerAnalysis_correlation_distinguish(Parameters* param, int8_t* (*f)(Parameters*));

//DPA
void differentialPowerAnalysis(Parameters* param, int8_t* (*f)(Parameters*));
void differentialPowerAnalysis_correlation_distinguish(Parameters* param, int8_t* (*f)(Parameters*));

#endif


