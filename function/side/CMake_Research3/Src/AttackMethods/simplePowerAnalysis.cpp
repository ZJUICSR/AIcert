#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <thread>
#include <time.h>
#include "../../Inc/side_channel_attack_methods.h"
#include "../../Inc/TRACE/Trace.h"
#include "../../Inc/TRACE/TraceTool.h"
#include "../../Inc/BASETOOL/BaseTools.h"
#include "../../Inc/CNNModel/cnn.h"

using namespace std;

#if ISTHREAD

/***********宏定义区*************/
#define COV1_TEMPLATE_LEN 280
#define COV2_TEMPLATE_LEN 352
#define RELU_TEMPLATE_LEN 2000
#define POOL_TEMPLATE_LEN 2500
#define CONN_TEMPLATE_LEN 4400

#define POINT_NUM 10000000

#define COV2_GAP 4000
#define RELU_GAP 2000
#define POOL_GAP 10000
/********************************/
enum SUB_OP_TYPE {COV1=0,COV2=1,RULE=2,POOL=3,CONN=4};
typedef struct Area
{
  SUB_OP_TYPE type = COV1;   //0=cov1第一种卷积; 1=cov2第二种卷积; 2=relu激活函数; 3=pool池化; 4=conn全连接,不同type的值对应不同的层
  int start = 0;  //该层的起始点
  int length = 0; //该层一共占用多少个点
};
class SpaAI
{
private:
    /* data */
public:
    SpaAI(/* args */);
    ~SpaAI();
    void createTemplate_COV1();

    void matchTemplate(TrsData* TrsData_result, const TrsData TrsData_Attack, const int startPoint_attack, const int length_attack, const TrsData TrsData_template, const int startPoint_template, const int length_template, float& max);
    void matchTemplate_thread(TrsData* TrsData_result, const TrsData TrsData_Attack, const int startPoint_attack, const int length_attack, const TrsData TrsData_template, const int startPoint_template, const int length_template, float& max);
};


SpaAI::SpaAI(/* args */)
{
}

SpaAI::~SpaAI()
{
}

void SpaAI::matchTemplate(TrsData* TrsData_result, const TrsData TrsData_Attack, const int startPoint_attack, const int length_attack, const TrsData TrsData_template, const int startPoint_template, const int length_template, float& max){

    for (int i = 0; i < length_attack-length_template; i++)
    {
        TrsData_result->samples[i+startPoint_attack]=abs(BaseTools::correlate(TrsData_template.samples+startPoint_template,TrsData_Attack.samples+i+startPoint_attack,length_template));
        if (max<TrsData_result->samples[i])
        {
            max = TrsData_result->samples[i];
        }
    }

}

void SpaAI::matchTemplate_thread(TrsData* TrsData_result, const TrsData TrsData_Attack, const int startPoint_attack, const int length_attack, const TrsData TrsData_template, const int startPoint_template, const int length_template, float& max){

    vector<std::thread> mythreads;
    int threadNum = 2;
    float* maxValue = new float[threadNum];
    int length = (length_attack-length_template)/threadNum;
    for (int threadIndex = 0; threadIndex < threadNum; threadIndex++)
    {
        int index = threadIndex * length;
        mythreads.push_back(std::thread(&SpaAI::matchTemplate,this,TrsData_result,TrsData_Attack,startPoint_attack+index,length,TrsData_template,startPoint_template,length_template,std::ref(maxValue[threadIndex])));
    }
    for (auto iter = mythreads.begin(); iter < mythreads.end(); iter++)
    {
        iter->join();
    }
    for (int i = 0; i < threadNum; i++)
    {
        if (max < maxValue[i])
        {
            max = maxValue[i];
        }
    }
    std::cout << "finish!" <<endl;

}

#define TEMPLATE_NUM 5 
#define SCALE 5

//是否采用平均值取点压缩模式
// #define AVE_SCALE
//是否开启多线程
#define TEST_THREAD
#define THREAD_NUM 5
void simplePowerAnalysis(Parameters* param)
{
    
    string inFile = (string)param->getSampleFile();
    string outFile = (string)param->getOutFile();
//压缩曲线
#if 0
    //读取模板文件
    string fileName_temp[TEMPLATE_NUM+s1] = {"./template/cov1_temp.trs","./template/cov2_temp.trs","./template/relu3_temp.trs","./template/pool3_temp.trs","./template/conn_temp.trs","./file/50-all-1.trs"};
    #ifdef AVE_SCALE
    string outfileName_temp[TEMPLATE_NUM+1] = { "./template/ave_scale"+to_string(SCALE)+"_cov1_temp.trs",
                                                "./template/ave_scale"+to_string(SCALE)+"_cov2_temp.trs",
                                                "./template/ave_scale"+to_string(SCALE)+"_relu3_temp.trs",
                                                "./template/ave_scale"+to_string(SCALE)+"_pool3_temp.trs",
                                                "./template/ave_scale"+to_string(SCALE)+"_conn_temp.trs",
                                                "./file/scale"+to_string(SCALE)+"_50-all-1.trs"};
    #else
    string outfileName_temp[TEMPLATE_NUM+1] = {"./template/scale"+to_string(SCALE)+"_cov1_temp.trs",
                                            "./template/scale"+to_string(SCALE)+"_cov2_temp.trs",
                                            "./template/scale"+to_string(SCALE)+"_relu3_temp.trs",
                                            "./template/scale"+to_string(SCALE)+"_pool3_temp.trs",
                                            "./template/scale"+to_string(SCALE)+"_conn_temp.trs",
                                            "./file/scale"+to_string(SCALE)+"_50-all-1.trs"};
    #endif
    TrsData trsData_temp[TEMPLATE_NUM+1],outTrsData_temp[TEMPLATE_NUM+1];
    Trace* trace_temp[TEMPLATE_NUM+1];
    for (int i = 0; i < TEMPLATE_NUM+1; i++)
    {
        trace_temp[i] = new Trace(fileName_temp[i]);
        trace_temp[i]->readNext(&trsData_temp[i]);
        outTrsData_temp[i].samples = new float[trace_temp[i]->trsHead.NS/SCALE];
    }
    
    for (int i = 0; i < TEMPLATE_NUM+1; i++)
    {
        for (int j = 0; j < trace_temp[i]->trsHead.NS/SCALE; j++)
        {
            #ifdef AVE_SCALE
            float sum = 0;
            for (int k = 0; k < SCALE; k++)
            {
                sum += trsData_temp[i].samples[j*SCALE+k];
            }
            outTrsData_temp[i].samples[j] = sum/SCALE;
            #else
            outTrsData_temp[i].samples[j] = trsData_temp[i].samples[j*SCALE];;
            #endif
        }
        //保存结果
        TrsHead trsHead_result = trace_temp[i]->trsHead;
        Trace *trace_result = new Trace();
        /*********头部信息*********/
        trsHead_result.NT = 1;
        trsHead_result.NS = trsHead_result.NS/SCALE;
        trsHead_result.DS = 0;
        trsHead_result.YS = 1;
        trsHead_result.SC = 0x14;//float存储类型
        trsHead_result.GT_length = 0;
        trsHead_result.DC_length = 0;
        trsHead_result.XL_length = 0;
        trsHead_result.YL_length = 0;
        trsHead_result.TS = 0;
        trace_result->createTrace(outfileName_temp[i].data(), &trsHead_result, &outTrsData_temp[i]);
        delete trace_result;
    }
#endif

//分析曲线
#if 1
    clock_t start,readEnd[2],end[TEMPLATE_NUM];
    start = clock();
    //读取待攻击文件
    string fileName_attack = inFile;
    TrsData trsData_attack[TEMPLATE_NUM];
    Trace *trace_attack = new Trace(fileName_attack);
    trace_attack->readNext(&trsData_attack[0]);
    for (int i = 1; i < TEMPLATE_NUM; i++)
    {
        trsData_attack[i].samples = new float[trace_attack->trsHead.NS];
        memcpy(trsData_attack[i].samples,trsData_attack[0].samples,sizeof(float)*trace_attack->trsHead.NS);
    }
    readEnd[0] = clock();
    //读取模板文件
    string fileName_temp[TEMPLATE_NUM] = {"./Trs/samples/spa/template/cov1_temp.trs","./Trs/samples/spa/template/cov2_temp.trs","./Trs/samples/spa/template/relu3_temp.trs","./Trs/samples/spa/template/pool3_temp.trs","./Trs/samples/spa/template/conn_temp.trs",};
    // string fileName_temp[TEMPLATE_NUM] = {"./template/scale_cov1_temp.trs","./template/scale_cov2_temp.trs","./template/scale_relu3_temp.trs","./template/scale_pool3_temp.trs","./template/scale_conn_temp.trs"};
    // string fileName_temp[TEMPLATE_NUM] = {  "./Trs/samples/spa/template/scale"+to_string(SCALE)+"_cov1_temp.trs",
    //                                         "./Trs/samples/spa/template/scale"+to_string(SCALE)+"_cov2_temp.trs",
    //                                         "./Trs/samples/spa/template/scale"+to_string(SCALE)+"_relu3_temp.trs",
    //                                         "./Trs/samples/spa/template/scale"+to_string(SCALE)+"_pool3_temp.trs",
    //                                         "./Trs/samples/spa/template/scale"+to_string(SCALE)+"_conn_temp.trs"    };
    TrsData trsData_temp[TEMPLATE_NUM];
    Trace* trace_temp[TEMPLATE_NUM];
    for (int i = 0; i < TEMPLATE_NUM; i++)
    {
        trace_temp[i] = new Trace(fileName_temp[i]);
        trace_temp[i]->readNext(&trsData_temp[i]);
    }
    //结果缓存
    TrsData* trsData_result[TEMPLATE_NUM];
    for (int i = 0; i < TEMPLATE_NUM; i++)
    {
        trsData_result[i] = new TrsData;
        trsData_result[i]->samples = new float[trace_attack->trsHead.NS];
    }
    int length[TEMPLATE_NUM] = {(COV1_TEMPLATE_LEN / SCALE),
                                (COV2_TEMPLATE_LEN / SCALE),
                                (RELU_TEMPLATE_LEN / SCALE),
                                (POOL_TEMPLATE_LEN / SCALE),
                                (CONN_TEMPLATE_LEN / SCALE)};
    //保存结果
    TrsHead trsHead_result = trace_attack->trsHead;
    // Trace *trace_result = new Trace();
    Trace trace_result;
	/*********头部信息*********/
	trsHead_result.NT = 1;
	trsHead_result.DS = 0;
	trsHead_result.YS = 1;
	trsHead_result.SC = 0x14;//float存储类型
	trsHead_result.GT_length = 0;
	trsHead_result.DC_length = 0;
	trsHead_result.XL_length = 0;
	trsHead_result.YL_length = 0;
	trsHead_result.TS = 0;
    readEnd[1] = clock();
    SpaAI sapAI;
    float max[TEMPLATE_NUM] = {0};
    // string outfileName[TEMPLATE_NUM] = {"./Outfile/result_cov1.trs","./Outfile/result_cov2.trs","./Outfile/result_relu.trs","./Outfile/result_pool.trs","./Outfile/result_conn.trs"};
    #ifdef AVE_SCALE
    string outfileName[TEMPLATE_NUM] = {"./Outfile/result_ave_scale"+to_string(SCALE)+"_cov1.trs",
                                        "./Outfile/result_ave_scale"+to_string(SCALE)+"_cov2.trs",
                                        "./Outfile/result_ave_scale"+to_string(SCALE)+"_relu.trs",
                                        "./Outfile/result_ave_scale"+to_string(SCALE)+"_pool.trs",
                                        "./Outfile/result_ave_scale"+to_string(SCALE)+"_conn.trs"};
    #else
    string outfileName[TEMPLATE_NUM] = {outFile + "result_scale"+to_string(SCALE)+"_cov1.trs",
                                        outFile + "result_scale"+to_string(SCALE)+"_cov2.trs",
                                        outFile + "result_scale"+to_string(SCALE)+"_relu.trs",
                                        outFile + "result_scale"+to_string(SCALE)+"_pool.trs",
                                        outFile + "result_scale"+to_string(SCALE)+"_conn.trs"};

    
    #endif
    vector<std::thread> mythreads;
    // for (int i = 0; i < THREAD_NUM; i++)
    for (int i = 0; i < THREAD_NUM; i++)
    {
        #ifdef TEST_THREAD 
            mythreads.push_back(std::thread(&SpaAI::matchTemplate_thread, &sapAI, trsData_result[i],trsData_attack[i],0,trace_attack->trsHead.NS,trsData_temp[i],0,length[i], std::ref(max[i])));
        #else
        max[i] = sapAI.matchTemplate(trsData_result[i],trsData_attack[i],0,trace_attack->trsHead.NS,trsData_temp[i],0,length[i]);
        // cout << "max[i] = " << max[i] <<endl;
        // for (int j = 0; j < trace_attack->trsHead.NS-length[i]; j++)
        // {
        //     if (max[i] *0.75 > trsData_result[i]->samples[j])
        //     {
        //         trsData_result[i]->samples[j] = 0;
        //     }
        // }
        trsHead_result.NS = trace_attack->trsHead.NS - length[i];
        trace_result.createTrace(outfileName[i].data(), &trsHead_result, trsData_result[i]);
        end[i] = clock();
        #endif
    }
    #ifdef TEST_THREAD 
    for (auto iter = mythreads.begin(); iter < mythreads.end(); iter++)
    {
        iter->join();
    }
    for (int i = THREAD_NUM; i < TEMPLATE_NUM; i++)
    {
        sapAI.matchTemplate_thread(trsData_result[i],trsData_attack[i],0,trace_attack->trsHead.NS,trsData_temp[i],0,length[i],std::ref(max[i]));
    }
    for (int i = 0; i < TEMPLATE_NUM; i++)
    {
        trace_result.createTrace(outfileName[i].data(), &trsHead_result, trsData_result[i]);
    }
    end[TEMPLATE_NUM-1] = clock();
    #else
    for (int i = THREAD_NUM; i < TEMPLATE_NUM; i++)
    {
        sapAI.matchTemplate(trsData_result[i],trsData_attack[i],0,trace_attack->trsHead.NS,trsData_temp[i],0,length[i]);
    }
    for (int i = 0; i < TEMPLATE_NUM; i++)
    {
        trace_result.createTrace(outfileName[i].data(), &trsHead_result, trsData_result[i]);
    }
    end[TEMPLATE_NUM-1] = clock();
    #endif
    //输出程序中各个子操作的耗时
    std::cout << "Executing the program takes " << 1000*(end[TEMPLATE_NUM-1]-start)/CLOCKS_PER_SEC << "ms." << endl;
    // std::cout << "Executing the program takes " << 1000*(end[TEMPLATE_NUM-1]-start)/CLOCKS_PER_SEC << "ms. Where : " << endl;
    // std::cout << "Reading the attack file take " << 1000*(readEnd[0]-start)/CLOCKS_PER_SEC << "ms." << endl;
    // std::cout << "Reading the template file take " << 1000*(readEnd[1]-readEnd[0])/CLOCKS_PER_SEC << "ms." << endl;
    // std::cout << "Matching COV1 program take " << 1000*(end[0]-readEnd[1])/CLOCKS_PER_SEC << "ms." << endl;
    // std::cout << "Matching COV2 program take " << 1000*(end[1]-end[0])/CLOCKS_PER_SEC << "ms." << endl;
    // std::cout << "Matching RELU program take " << 1000*(end[2]-end[1])/CLOCKS_PER_SEC << "ms." << endl;
    // std::cout << "Matching POOL program take " << 1000*(end[3]-end[2])/CLOCKS_PER_SEC << "ms." << endl;
    // std::cout << "Matching CONN program take " << 1000*(end[4]-end[3])/CLOCKS_PER_SEC << "ms." << endl;
#endif
}

#else

/***********宏定义区*************/
#define COV1_TEMPLATE_LEN 280
#define COV2_TEMPLATE_LEN 352
#define RELU_TEMPLATE_LEN 2000
#define POOL_TEMPLATE_LEN 2500
#define CONN_TEMPLATE_LEN 4400

#define POINT_NUM 10000000

#define COV2_GAP 4000
#define RELU_GAP 2000
#define POOL_GAP 10000
/********************************/
using namespace std;
typedef struct Area
{
  int type = 0;   //0=cov1第一种卷积; 1=cov2第二种卷积; 2=relu激活函数; 3=pool池化; 4=conn全连接,不同type的值对应不同的层
  int start = 0;  //该层的起始点
  int length = 0; //该层一共占用多少个点
};
class Spa
{
public:
  /*
 * filein:被攻击曲线trs文件的路径
 * result:神经网络每一层结构信息的指针（一层结构对应一个Area）
 */
  static void spa(const char *filein, Area *result);

private:
  static void gen_base(const char *filein, const char *fileout, int nt, int ns, int immp);
  static void gen_subcov1(Area *cov1Area);
  static void gen_subrelu(Area *reluArea);
  static void gen_subpool(Area *reluArea);
  static void gen_subcov2(Area *cov2Area);
  static void gen_subconn(Area *connArea);
};

void Spa::gen_base(const char *filein, const char *fileout, int nt, int ns, int immp)
{
    Trace *trace_base = new Trace(filein);
    TrsData *trsData_base = new TrsData;

    TrsData trsData_result;
    trsData_result.samples = new float[ns];
    for (int i = 0; i < nt; i++)
    {
        trace_base->readIndexTrace(trsData_base, i);
        for (int j = 0; j < ns; j++)
        {
            if (i == (nt - 0))
            {
                trsData_result.samples[j] = trsData_base->samples[immp + j];
                trsData_result.samples[j] /= nt;
            }
            else
            {
                trsData_result.samples[j] = trsData_base->samples[immp + j];
            }
        }
    }

    TrsHead trsHead_base = trace_base->trsHead;
    trsHead_base.NT = 1;
    trsHead_base.NS = ns;
    trsHead_base.DS = 0;
    trsHead_base.YS = 1;
    trsHead_base.SC = 0x14;
    trsHead_base.GT_length = 0;
    trsHead_base.DC_length = 0;
    trsHead_base.XL_length = 0;
    trsHead_base.YL_length = 0;
    trsHead_base.TS = 0;
    Trace trace;

    trace.createTrace(fileout, &trsHead_base, &trsData_result);
}

void gen_outfile_cov1(TrsData *trsData_attack)
{
    Trace *trace_base_cov1 = new Trace("./Base/cov1_temp.trs");
    TrsData *trsData_cov1_base = new TrsData;
    trace_base_cov1->readIndexTrace(trsData_cov1_base, 0);
    TraceTools::corr(trsData_cov1_base, trsData_attack, 10000000, COV1_TEMPLATE_LEN, 0, trace_base_cov1->trsHead, "./Outfile/outfile_cov1.trs");
}

void gen_outfile_relu(TrsData *trsData_attack)
{
    Trace *trace_base_relu3 = new Trace("./Base/relu3_temp.trs");
    TrsData *trsData_relu3_base = new TrsData;
    trace_base_relu3->readIndexTrace(trsData_relu3_base, 0);
    TraceTools::corr(trsData_relu3_base, trsData_attack, 10000000, RELU_TEMPLATE_LEN, 0, trace_base_relu3->trsHead, "./Outfile/outfile_relu3.trs");
}

void gen_outfile_pool(TrsData *trsData_attack)
{
    Trace *trace_base_pool3 = new Trace("./Base/pool3_temp.trs");
    TrsData *trsData_pool3_base = new TrsData;
    trace_base_pool3->readIndexTrace(trsData_pool3_base, 0);
    TraceTools::corr(trsData_pool3_base, trsData_attack, 10000000, POOL_TEMPLATE_LEN, 0, trace_base_pool3->trsHead, "./Outfile/outfile_pool3.trs");
}

void gen_outfile_cov2(TrsData *trsData_attack)
{
    Trace *trace_base_cov2 = new Trace("./Base/cov2_temp.trs");
    TrsData *trsData_cov2_base = new TrsData;
    trace_base_cov2->readIndexTrace(trsData_cov2_base, 0);
    TraceTools::corr_78(trsData_cov2_base, trsData_attack, 10000000, COV2_TEMPLATE_LEN, 0, trace_base_cov2->trsHead, "./Outfile/outfile_cov2.trs");
}

void gen_outfile_conn(TrsData *trsData_attack)
{
    Trace *trace_base_conn = new Trace("./Base/conn_temp.trs");
    TrsData *trsData_conn_base = new TrsData;
    trace_base_conn->readIndexTrace(trsData_conn_base, 0);
    TraceTools::corr_9(trsData_conn_base, trsData_attack, 10000000, CONN_TEMPLATE_LEN, 0, trace_base_conn->trsHead, "./Outfile/outfile_conn.trs");
}


void Spa::gen_subcov1(Area *cov1Area)
{
    Trace *trace_test = new Trace("./Outfile/outfile_cov1.trs");
    TrsData *trsData_test = new TrsData;
    trace_test->readIndexTrace(trsData_test, 0);
    int flag = 1;
    for (int i = 0; i < POINT_NUM - COV1_TEMPLATE_LEN; i++)
    {
        if (trsData_test->samples[i] > 0.7)
        {
            if (flag)
            {
                cov1Area[0].start = i;
                cov1Area[0].type = 0;
                flag = 0;
            }
            cov1Area[0].length = i - cov1Area[0].start + COV1_TEMPLATE_LEN;
        }
    }
}

void Spa::gen_subrelu(Area *reluArea)
{
    Trace *trace_test = new Trace("./Outfile/outfile_relu3.trs");
    TrsData *trsData_test = new TrsData;
    trace_test->readIndexTrace(trsData_test, 0);
    int flag = 0, point = 0;
    //判断第1层激活函数
    for (int i = 0; i < POINT_NUM - RELU_TEMPLATE_LEN; i++)
    {
        if (trsData_test->samples[i] > 0.5)
        {
            if (flag == 0)
            {
                reluArea[0].start = i;
                reluArea[0].type = 2;
                flag = 1;
            }
            point = i;
        }
        if ((i - point) > RELU_GAP && flag == 1)
        {
            reluArea[0].length = i - reluArea[0].start - RELU_GAP + RELU_TEMPLATE_LEN;
            flag = 0;
            break;
        }
    }
    //判断第2层激活函数
    for (int i = reluArea[0].start + reluArea[0].length; i < POINT_NUM - RELU_TEMPLATE_LEN; i++)
    {
        if (trsData_test->samples[i] > 0.5)
        {
            if (flag == 0)
            {
                reluArea[1].start = i;
                reluArea[1].type = 2;
                flag = 1;
            }
            point = i;
        }
        if ((i - point) > RELU_GAP && flag == 1)
        {
            reluArea[1].length = i - reluArea[1].start - RELU_GAP + RELU_TEMPLATE_LEN;
            flag = 0;
            break;
        }
    }
    //判断第3层激活函数
    for (int i = reluArea[1].start + reluArea[1].length; i < POINT_NUM - RELU_TEMPLATE_LEN; i++)
    {
        if (trsData_test->samples[i] > 0.5)
        {
            if (flag == 0)
            {
                reluArea[2].start = i;
                reluArea[2].type = 2;
                flag = 1;
            }
            point = i;
        }
        if ((i - point) > RELU_GAP && flag == 1)
        {
            reluArea[2].length = i - reluArea[2].start - RELU_GAP + RELU_TEMPLATE_LEN;
            flag = 0;
            break;
        }
    }
}

void Spa::gen_subpool(Area *reluArea)
{
    Trace *trace_test = new Trace("./Outfile/outfile_pool3.trs");
    TrsData *trsData_test = new TrsData;
    trace_test->readIndexTrace(trsData_test, 0);
    int flag = 0, point = 0;
    //判断池化1
    for (int i = 0; i < POINT_NUM - POOL_TEMPLATE_LEN; i++)
    {
        if (trsData_test->samples[i] > 0.09)
        {
            if (flag == 0)
            {
                reluArea[0].start = i;
                reluArea[0].type = 3;
                flag = 1;
            }
            point = i;
        }
        if ((i - point) > POOL_GAP && flag == 1)
        {
            reluArea[0].length = i - reluArea[0].start - POOL_GAP + POOL_TEMPLATE_LEN;
            flag = 0;
            break;
        }
    }
    //判断池化2
    for (int i = reluArea[0].start + reluArea[0].length; i < POINT_NUM - POOL_TEMPLATE_LEN; i++)
    {
        if (trsData_test->samples[i] > 0.11)
        {
            if (flag == 0)
            {
                reluArea[1].start = i;
                reluArea[1].type = 3;
                flag = 1;
            }
            point = i;
        }
        if ((i - point) > POOL_GAP && flag == 1)
        {
            reluArea[1].length = i - reluArea[1].start - POOL_GAP + POOL_TEMPLATE_LEN;
            flag = 0;
            break;
        }
    }
    //判断池化3
    for (int i = reluArea[1].start + reluArea[1].length; i < POINT_NUM - POOL_TEMPLATE_LEN; i++)
    {
        if (trsData_test->samples[i] > 0.13)
        {
            if (flag == 0)
            {
                reluArea[2].start = i;
                reluArea[2].type = 3;
                flag = 1;
            }
            point = i;
        }
        if ((i - point) > POOL_GAP && flag == 1)
        {
            reluArea[2].length = i - reluArea[2].start - POOL_GAP + POOL_TEMPLATE_LEN;
            flag = 0;
            break;
        }
    }
}

void Spa::gen_subcov2(Area *cov2Area)
{
    Trace *trace_test = new Trace("./Outfile/outfile_cov2.trs");
    TrsData *trsData_test = new TrsData;
    trace_test->readIndexTrace(trsData_test, 0);
    int flag = 0, point = 0;
    //判断第2层卷积
    for (int i = 0; i < POINT_NUM - COV2_TEMPLATE_LEN; i++)
    {
        if (trsData_test->samples[i] > 0.4)
        {
            if (flag == 0)
            {
                cov2Area[0].start = i;
                cov2Area[0].type = 1;
                flag = 1;
            }
            point = i;
        }
        if ((i - point) > COV2_GAP && flag == 1)
        {
            cov2Area[0].length = i - cov2Area[0].start - COV2_GAP + COV2_TEMPLATE_LEN;
            flag = 0;
            break;
        }
    }
    //判断第3层卷积
    for (int i = cov2Area[0].start + cov2Area[0].length; i < POINT_NUM - COV2_TEMPLATE_LEN; i++)
    {
        if (trsData_test->samples[i] > 0.4)
        {
            if (flag == 0)
            {
                cov2Area[1].start = i;
                cov2Area[1].type = 1;
                flag = 1;
            }
            point = i;
        }
        if ((i - point) > COV2_GAP && flag == 1)
        {
            cov2Area[1].length = i - cov2Area[1].start - COV2_GAP + COV2_TEMPLATE_LEN;
            flag = 0;
            break;
        }
    }
}

void Spa::gen_subconn(Area *connArea)
{
    Trace *trace_test = new Trace("./Outfile/outfile_conn.trs");
    TrsData *trsData_test = new TrsData;
    trace_test->readIndexTrace(trsData_test, 0);
    int flag = 1;
    for (int i = 0; i < POINT_NUM - CONN_TEMPLATE_LEN; i++)
    {
        if (trsData_test->samples[i] > 0.6)
        {
            if (flag)
            {
                connArea[0].start = i;
                connArea[0].type = 4;
                flag = 0;
            }
            connArea[0].length = i - connArea[0].start + CONN_TEMPLATE_LEN;
        }
    }
}

void Spa::spa(const char *filein, Area *result)
{
    string command;
    command = "mkdir Outfile ";
    system(command.c_str());

    Trace *trace_attack = new Trace(filein);
    TrsData *trsData_attack = new TrsData;
    trace_attack->readIndexTrace(trsData_attack, 0);

    thread t1(gen_outfile_cov1, trsData_attack);
    thread t2(gen_outfile_relu, trsData_attack);
    thread t3(gen_outfile_pool, trsData_attack);
    thread t4(gen_outfile_cov2, trsData_attack);
    thread t5(gen_outfile_conn, trsData_attack);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();

    Spa::gen_subcov1(result + 0);
    Spa::gen_subrelu(result + 1);
    Spa::gen_subpool(result + 4);
    Spa::gen_subcov2(result + 7);
    Spa::gen_subconn(result + 9);
}

#endif

