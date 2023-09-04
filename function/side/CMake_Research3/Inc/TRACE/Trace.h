#pragma once
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;
enum SAMPLE_TYPE
{
	BYTE = 0,
	SHORT = 1,
	INT = 2,
	FLOAT = 3
};
#define SC_TYPE(d) ((d==0x14)?FLOAT :((d == 0x04)? INT :((d==0x02)? SHORT : BYTE)))
struct TrsData {
	uint8_t* TSData = nullptr;	//曲线title			读取顺序：1
	uint8_t* data = nullptr;	//明文数据+密文数据	读取顺序：2
	float* samples = nullptr;	//功耗曲线			读取顺序：3
};
struct TrsHead
{
	int NT = 0;			// Number of traces
	int NS = 0;			// Number of samples per trace
	int8_t SC = 0;		// Sample coding (e.g. type and length in bytes of each sample)
	short DS = 0;		// Length of cryptographic data included in trace
	uint8_t TS = 0;		// Title space reserved per trace
	uint8_t* GT = nullptr;	// Global trace title
	uint8_t GT_length = 0;//length of Global trace title
	uint8_t* DC = nullptr;	// Description
	uint8_t DC_length = 0;//length of Description
	int XO = 0;			// Offset in X-axis for trace representation
	uint8_t* XL = nullptr;	// Label of X-axis
	uint8_t XL_length = 0;//length of Label of X-axis
	uint8_t* YL = nullptr;	// Label of Y-axis
	uint8_t YL_length = 0;//length of Label of Y-axis
	float XS = 0;		// Scale value for X-axis
	float YS = 0;		// Scale value for Y-axis
	int TO = 0;			// Trace offset for displaying trace numbers
	uint8_t LS = 0;		// Logarithmic scale
	float RG = 0;		//Range of the scope used to perform acquisition
	int CL = 0;			//Coupling of the scope used to perform acquisition
	float OS = 0;		//Offset of the scope used to perform acquisition
	float II = 0;		//Input impedance of the scope used to perform acquisition
	uint8_t* AI = nullptr;	//Device ID of the scope used to perform acquisition
	int FT = 0;			//The type of filter used during acquisition
	float FF = 0;		//Frequency of the filter used during acquisition
	float FR = 0;		//Range of the filter used during acquisition
	bool EU = 0;		//External clock used
	float ET = 0;		//External clock threshold
	int EM = 0;			//External clock multiplier
	int EP = 0;			//External clock phase shift
	int ER = 0;			//External clock resampler mask
	bool RE = 0;		//External clock resampler enabled
	float EF = 0;		//External clock frequency
	int EB = 0;			//External clock time base
	//int TB;		// Trace block marker: an empty TLV that marks the end of the header
};
struct Trs
{
	TrsHead trsHead;
	TrsData* trsData;
};
class Trace
{
	//变量
public:
	int inDataLen;		//加密/解密前的数据长度
	int outDataLen;		//加密/解密后的数据长度
	int currentTrace;	//当前读取的波形
	TrsHead trsHead;	//Trace head
private:
	ifstream infile;//读取文件

//第一套读取函数，不需要输入文件作为参数（通过成员变量控制：需先使用Trace(const char* file)构造Trace对象）。
public:
	Trace();
	Trace(const char* file);
	//重载构造函数
	Trace(const char* file, int inDataLen, int outDataLen);
	//读取全部数据
	void readAllTrace(TrsData* trsData);//trsData：返回的数据
	//读取单条数据：一条一条读取
	void readNext(TrsData* trsData);//trsData：返回的数据
	//读取指定下标单条数据
	void readIndexTrace(TrsData* trsData,int index);//trsData：返回的数据

//用于创建和删除
public:
	//创建trs文件->缺，待补
	void createTrace(const char* filename, TrsHead* trsHead, TrsData* trsData);
	//删除指定曲线(建议输出文件和输入文件不要取同一个名字)
	//void deleteTrace(const char* outFileName, const char* inFileName, int* trsIndexArray, int deleteNum);
	//保存指定曲线(建议输出文件和输入文件不要取同一个名字)
	void saveTraceArea(const char* outFileName, const char* inFileName, int startPoint, int length);

//第二套读取函数，需要输入文件作为参数（不需要Trace对象）
public:
	//读取头文件(输出到trsHead)
	static void readHeard(ifstream* inFile, TrsHead* trsHead);
	//读取单条数据(一条一条读取);返回的数据:trsData
	static void readNext(ifstream* inFile, TrsData* trsData, const TrsHead trsHead, int* currentTrace);
	/*
	* inFile:文件来源（注意是文件指针参数）
	* trsData：输出数据
	* trsHead：读取数据所需的头部信息
	* index：读取第几条数据
	* currentTrace：当前文件处于第几条
	*/
	//读取指定下标单条数据
	static void readIndexTrace(ifstream* inFile, TrsData* trsData, const TrsHead trsHead, int index, int* currentTrace);//trsData：返回的数据
	//写入单条曲线（配合writeHead使用，先单条写入，最后在文件的前面写入头部信息）
	static void writeNext(ofstream* outFile, TrsData* trsData, const TrsHead trsHead);
	//写入头部信息（配合writeNext使用，在文件的前面写入头部信息，默认该输出文件没有头部信息）
	static void writeHead(ofstream* outFile, const TrsHead trsHead);
	//析构函数
	~Trace();
private:
	//打开trs文件
	void readHeard(const char* file);
};
