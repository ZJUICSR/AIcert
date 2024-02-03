#include<stdio.h>
#include <cstring>
#include <math.h>
#include "../Inc/TRACE/Trace.h"
#include "../Inc/BASETOOL/BaseTools.h"
#include "../Inc/CNNModel/arm_nnexamples_cifar10_inputs.h"

double gaussrand(double V)
{
    static double V1, V2, S;
    static int phase = 0;
    double X;
    double E = 0;//期望
    if (phase == 0) {
    do {
    double U1 = (double)rand() / RAND_MAX;
    double U2 = (double)rand() / RAND_MAX;

    V1 = 2 * U1 - 1;
    V2 = 2 * U2 - 1;
    S = V1 * V1 + V2 * V2;
    } while (S >= 1 || S == 0);
    X = V1 * sqrt(-2 * log(S) / S);
    }
    else
    X = V2 * sqrt(-2 * log(S) / S);
    phase = 1 - phase;
    return X = X * V + E;
}

void printTrs(char* trs){
    Trace trace(trs);
    TrsData trsData;
    trace.readIndexTrace(&trsData, 0);
    for(int i = 0; i < 1000; i++){
        printf("%f ", trsData.samples[i]);
    }
    free(trsData.samples);
    free(trsData.data);
    free(trsData.TSData);
}

void repairTrs(char* trsSrc, char* trsDis){

    //具体问题具体修改数据
    int NS = 10000;
    int NT = 10000;
    int DS = 3072;

    Trace trace(trsSrc);
    TrsData trsDataS;
    TrsData trsData;
    trsData.samples=new float[NS];
    trsData.data=new uint8_t[DS];

    ofstream* outfile=new ofstream();   
    outfile->open(trsDis, ios::out | ios::binary | ios::trunc);

    TrsHead trsHead_result;
    trsHead_result.NT = NT;
    trsHead_result.NS = NS;
    trsHead_result.DS = DS;
    trsHead_result.YS = 1;
    trsHead_result.SC = 0x14;//float存储类型
    trsHead_result.GT_length = 0;
    trsHead_result.DC_length = 0;
    trsHead_result.XL_length = 0;
    trsHead_result.YL_length = 0;
    trsHead_result.TS = 0;
    Trace::writeHead(outfile, trsHead_result);

    for(int i =0 ;i<trsHead_result.NT;i++){
        trace.readIndexTrace(&trsDataS,i);
        

        memcpy(trsData.samples, trsDataS.samples,trsHead_result.NS*sizeof(float));
        memcpy(trsData.data, trsDataS.data,trsHead_result.DS*sizeof(uint8_t)); 
        Trace::writeNext(outfile, &trsData, trsHead_result);
    }

    free(trsData.samples);
    free(trsData.data);

    outfile->close();
    free(outfile);

}

void trsSplice(char* trsOne, char* trsTwo, char* out){


    Trace trace1(trsOne);
    Trace trace2(trsTwo);

    // int DS = trace1.trsHead.DS;
    // int NS = trace1.trsHead.NS;

    TrsData trsDataIn;
    TrsData trsData;
    trsData.samples=(float*)malloc(trace1.trsHead.NS*sizeof(float));
    trsData.data=(uint8_t*)malloc(trace1.trsHead.DS*sizeof(uint8_t));

    ofstream* outfile=new ofstream();   
    outfile->open(out, ios::out | ios::binary | ios::trunc);

    TrsHead trsHead_result;
    trsHead_result.NT = trace1.trsHead.NT+trace2.trsHead.NT;
    trsHead_result.NS = trace1.trsHead.NS;
    trsHead_result.DS = trace1.trsHead.DS;
    trsHead_result.YS = 1;
    trsHead_result.SC = 0x14;//float存储类型
    trsHead_result.GT_length = 0;
    trsHead_result.DC_length = 0;
    trsHead_result.XL_length = 0;
    trsHead_result.YL_length = 0;
    trsHead_result.TS = 0;
    Trace::writeHead(outfile, trsHead_result);

    

    for(int i = 0; i<trace1.trsHead.NT; i++){
        trace1.readIndexTrace(&trsDataIn,i);
    
        memcpy(trsData.samples, trsDataIn.samples, trace1.trsHead.NS*sizeof(float));
        memcpy(trsData.data, trsDataIn.data,trace1.trsHead.DS*sizeof(uint8_t)); 
        Trace::writeNext(outfile, &trsData, trsHead_result);
    }
    
    for(int i =0; i< trace2.trsHead.NT; i++){
        trace2.readIndexTrace(&trsDataIn,i);
    
        memcpy(trsData.samples, trsDataIn.samples, trace2.trsHead.NS*sizeof(float));
        memcpy(trsData.data, trsDataIn.data,trace2.trsHead.DS*sizeof(uint8_t)); 
        Trace::writeNext(outfile, &trsData, trsHead_result);
    }
   
    // free(trsData.samples);
    // free(trsData.data);
    
    outfile->close();
    printf("done1\n");
    // free(outfile);

    
}

void fixattackindex(int attackIndex){
    printf("i:%d ", attackIndex/75);//
    printf("m:%d ", attackIndex%75/15);//
    printf("n:%d ", attackIndex%75%15/3);//
    printf("l:%d ", attackIndex%3);//
}

void cutOut(char* file_coor, char file_slit[], char file_out[], int threshold, int counts){
    // char file_cor[]="11/C_oneConv3.trs";
    // char file_slit[]="sparesult/layer1/conv/sout2/trace3.trs";
    // char file_out[]="nfmul/oneConv25.trs";

    // int index = 4;
    // float yuzhi = 0.6;

    int NS = 450;
 
    Trace trace_coor(file_coor);
    TrsData trsData_coor;
    Trace trace_slit(file_slit);
    TrsData trsData_slit;
// gaussrand(1)
    TrsData t;
    t.samples=(float*)malloc(NS*sizeof(float));
    // memset(t.samples, 0.0, NS*sizeof(float));
    t.data=(uint8_t*)malloc(trace_slit.trsHead.DS*sizeof(uint8_t));
    memset(t.data, 0, trace_slit.trsHead.DS*sizeof(uint8_t));//new
    
    // int counts = 0;

    ofstream* outfile=new ofstream();
    outfile->open(file_out, ios::out | ios::binary | ios::trunc); 
    TrsHead trsHead;
    trsHead.NT = trace_slit.trsHead.NT;
    trsHead.NS = NS;
    trsHead.DS = trace_slit.trsHead.DS;
    trsHead.YS = 1;
    trsHead.SC = 0x14;//float存储类型
    trsHead.GT_length = 0;
    trsHead.DC_length = 0;
    trsHead.XL_length = 0;
    trsHead.YL_length = 0;
    trsHead.TS = 0;
    Trace::writeHead(outfile,trsHead);

    // printf("done1\n");

    int index = 0;
    int startFlag = 0;
    int startIndex =  0;
    int endFlag = 0;
    int endIndex = 0; 

    for(int j = 0; j< trace_coor.trsHead.NT; j++){

        // printf("%d ", j);

        index = 0;
        startFlag = 0;
        startIndex =  0;
        endFlag = 0;
        endIndex = 0; 

        trace_coor.readIndexTrace(&trsData_coor, j);

        for(int i = 0; i< trace_coor.trsHead.NS; i++){
            
            if(startFlag == 0){
                if(trsData_coor.samples[i]>threshold){
                    startIndex = i;
                    startFlag = 1;
                    // endFlag = 0;
                }
            }

            if(endFlag == 0){
                if(trsData_coor.samples[i]<threshold){
                    endIndex = i;
                    endFlag = 1;
                    // startFlag = 0;
                }
            }

            if(startFlag & endFlag){
                index++;
                if(index = counts){
                    // printf("%d\n", endIndex - startIndex + 1);
                    // counts = index;
                    endFlag = 0;
                    startFlag = 0;
                    break;
                }
                
            }

        }

        trace_slit.readIndexTrace(&trsData_slit, j);
        memcpy(t.data, trsData_slit.data, trace_slit.trsHead.DS*sizeof(uint8_t));
        memset(t.samples, 0.0, NS*sizeof(float));
        memcpy(t.samples,trsData_slit.samples+startIndex,(endIndex - startIndex + 1)*sizeof(float));
        // printf("done1\n");
        // memset(t.samples+(endIndex - startIndex + 1), 0.0 ,(NS-(endIndex - startIndex + 1))*sizeof(float));
        // printf("done2\n");
        Trace::writeNext(outfile, &t,trsHead);

        
        
        
    }
    printf("done1\n");
    outfile->close();
    printf("done2\n");
    free(t.samples);
    free(t.data);
    
    free(outfile);

}

int main(int argc, char const *argv[])
{
    #if 0//打印trace

    printTrs("Trs/OcsilloscopeC1.trs");


    #elif 0//修复trace

    char* trsSrc = "Ocsilloscope.trs";
    char* trsDis = "Ocsilloscope_.trs";

    repairTrs(trsSrc, trsDis);
    repairTrs("OcsilloscopeC1.trs", "OcsilloscopeC1_.trs");
    

    #elif 0//拼接trace

    char* trsOne = "Ocsilloscope_55.trs";
    char* trsTwo = "Ocsilloscope_55(2).trs";
    char* trsOut = "Ocsilloscope_55_f.trs";

    trsSplice(trsOne, trsTwo, trsOut);
    
    #elif 0

    fixattackindex(21);

    #elif 1 //定位乘法 for different wts

    char file_coor[] = "OcsilloscopeC1_.trs";
    char file_slit[] = "Ocsilloscope_.trs";
    char file_out[] = "55_500M_1w.trs";
    int threshold = 0.1;
    int counts = 1;

    cutOut(file_coor, file_slit, file_out, threshold, counts);

    #elif 0 //测试bytetoH

    char str[2]="\0";
    uint8_t input[3072] = IMG_DATA;

    for(int i = 0; i<3072 ;i++){
        BaseTools::byteToH(input[i], str);
        for(int j = 0;j<2;j++){
            printf("%c", str[j]);
        }
        printf(" ");
        
    }
    


    #endif


    

    return 0;
}
