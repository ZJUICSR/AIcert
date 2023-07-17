#ifndef _SIDE_CHANNEL_ATTACKS_METHODS_H_
#define _SIDE_CHANNEL_ATTACKS_METHODS_H_
#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>
#include <cstring>
#include "../Inc/CNNModel/arm_nnexamples_cifar10_weights.h"
#include "../Inc/BASETOOL/BaseTools.h"


#define MIN_VALUE 1e-12
#define IS_DOUBLE_ZERO(value)  (std::abs(value) < MIN_VALUE)

#define HORMID 784//hor
#define HOR_TRS 9

#define SIZE 1024

inline void prints(char* str){
    int i = 0;
    while(str[i] != '\0'){
        printf("%c", str[i]);
        i++;
    }
}

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

    int midvalue_per_trace;

    int wt_for_whiteBoxTest;

}InParameters;

typedef struct{
    
    int forI;
    int forJ;
    int forK;
    int forM;
    int forN;
    int forL;
    uint8_t midHW;
    uint8_t** horMidHW;
    int horMidHWX;
    int horMidHWY;

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

    char* getSampleFile(){
        return in->samplesFile;
    };

    void setRandFile(char* fileName){
        strcpy(in->randFile, fileName);
    };

    char* getRandFile(){
        return in->randFile;
    };

    void setOutFile(char* fileName){
        strcpy(in->outFile, fileName);
    };

    char* getOutFile(){
        return in->outFile;
    };

    void setTraceNum(int i){
        in->trace_num = i;
    };

    int getTraceNum(){
        return in->trace_num;
    };

    void setAttackIndex(int i){
        in->attackindex = i;
    };

    int getAttackIndex(){
        return in->attackindex;
    };

    void setPointNumStart(int i){
        in->point_num_start = i;
        in->point_num = in->point_num_end - in->point_num_start + 1;
    };

    int getPointNumStart(){
        return in->point_num_start;
    };

    void setPointNumEnd(int i){
        in->point_num_end = i;
        in->point_num = in->point_num_end - in->point_num_start + 1;
    };

    int getPointNumEnd(){
        return in->point_num_end;
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

    void setHorMidHW(int indexX, int indexY, uint8_t value){
        mid->horMidHW[indexX][indexY] = value;
    }

    uint8_t getHorMidHW(int indexX, int indexY){
        return mid->horMidHW[indexX][indexY];
    }

    uint8_t* getHorMidHWXPointer(int i){
        return mid->horMidHW[i];
    }

    void setHorMidX(int i){
        mid->horMidHWX = i;
    }

    int getHorMidX(){
        return mid->horMidHWX;
    }

    void setHorMidHWY(int i){
        mid->horMidHWY = i;
    }

    int getHorMidHWY(){
        return mid->horMidHWY;
    }

    void setMidvaluePerTrace(int i){ //调用必须在setTraceNum()后
        in->midvalue_per_trace = i;
       
    }
    
    int getMidvaluePerTrace(){ //调用必须在setTraceNum()后
        return in->midvalue_per_trace;
    }

    void InitialHorMid(){
        mid->horMidHW = (uint8_t**)malloc(in->guess_size*sizeof(uint8_t*));
        for(int i =0 ;i < in->guess_size; i++){
                mid->horMidHW[i] = (uint8_t*)malloc(in->trace_num*in->midvalue_per_trace*sizeof(uint8_t));
                memset(mid->horMidHW[i], 0 ,in->trace_num*in->midvalue_per_trace*sizeof(uint8_t));
        }
    }
    
    void freeHorMidHW(){

        if(mid->horMidHW != NULL){
            for(int i =0 ; i < in->guess_size; i++){
                if(mid->horMidHW[i] != NULL)
                    free(mid->horMidHW[i]);
            }
            free(mid->horMidHW);
        }
        
    }

    void setWtForWhiteBoxTest(int i){
        in->wt_for_whiteBoxTest = i;
    }

    int getWtForWhiteBoxTest(){
        return in->wt_for_whiteBoxTest;
    }

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

    void setMidHW(uint8_t i){
        in->mid = i;
    };
    
    uint8_t getMidHW(){
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

        in->midvalue_per_trace = 0;

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
        mid->midHW = 0;
        mid->horMidHWX = 0;
        mid->horMidHWY = 0;
        mid->horMidHW = NULL;
    };
    //********END*********************************************************


    ~Parameters(){
        if(in != NULL) free(in);
        if(fParam != NULL) free(fParam);
        // freeHorMidHW();
        if(mid != NULL) free(mid);
    };

};

inline void selectParentheses(char* str, char* disStr){
    int strlen = 0;
    int flag = -1;
    int disStrlen = 0;
    while(str[strlen] != '\0'){
       
        if(str[strlen] == '('){
            flag = 0;
        }else if(str[strlen] == ')'){
            flag = 1;
        }

        if(flag == -1){
            strlen++;
            continue;
        }else if(flag == 1){
            disStr[disStrlen-1] = '\0';
            break;
        }else if(flag == 0){
            disStr[disStrlen] = str[strlen+1];
            disStrlen++;
        }

        strlen++;
        
    }
}

inline void selectParenthesesNum(char* str, int* num){//printf("done1\n");
    char disStr[SIZE] = "\0";
    selectParentheses(str, disStr);

    #if 0
    prints(disStr);
    

    #endif

    char tem[SIZE]="\0";
    int temlen = 0;
    int disStrlen = 0;
    int numlen = 0;
    while(disStr[disStrlen]!='\0'){

        if(disStr[disStrlen]!='.'){
            tem[temlen]=disStr[disStrlen];
            temlen++;
        }else{
            // printf("\n");
            // prints(tem);
            num[numlen]=BaseTools::charToNumD(tem);
            numlen++;
            strcpy(tem, "\0");
            temlen = 0;
            
        }

        disStrlen++;
    }
    tem[temlen]=disStr[disStrlen];
    num[numlen]=BaseTools::charToNumD(tem);
}

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

//HPA
void horizontalPowerAnalysis(Parameters* param, int8_t* (*f)(Parameters*));
void horizontalPowerAnalysis_correlation_distinguish(Parameters* param, int8_t* (*f)(Parameters*));

//TTEST
void ttest_non_specific(Parameters* param, int8_t* (*f)(Parameters*));
void X2_test(Parameters* param, int8_t* (*f)(Parameters*));

#endif


