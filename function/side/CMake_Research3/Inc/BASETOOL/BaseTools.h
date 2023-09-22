#pragma once
#include <stdint.h>

#include <vector>
typedef struct MaxCorrStrcut
{
	float corr = 0;
	int point = 0;
	int key = 0;
};

enum COMPARE_MODE
{
	DESCENDING_SORT = 0,
	ASCENDING_SORT = 1
};
static bool descendingOrder(const MaxCorrStrcut a, const MaxCorrStrcut b) { return a.corr < b.corr; };//����
static bool ascendingOrder(const MaxCorrStrcut a, const MaxCorrStrcut b) { return a.corr > b.corr; };//����

class BaseTools
{
public:
	
	//���㺺������
	static int hanmingWeight(const uint8_t data);
	static int hanmingWeight(const int8_t data);
	static int hanmingWeight(const int32_t data,int bias);
	static int hdistance( int x, int y);
	int hanmingWeight(const uint16_t data);
	int hanmingWeight(const uint32_t data);
	int hanmingWeight(const uint64_t data);
	static int hanmingWeight(const int32_t data);
	static double correlate(double* src1, double* src2, int len);
	static double correlate(float* src1, float* src2, int len);
	static double corr(uint8_t* hw_arrays, float* power, int len);
	static double diff(uint8_t* hw_arrays, float* power, int len, int thresholdL);
	static double ttest(float* sample1, float* sample2, int len1, int len2);
	static double x2test(uint8_t* hw_arrays, float* power, int len);
	//bool descendingOrder(const MaxCorrStrcut a, const MaxCorrStrcut b);//����
	//bool ascendingOrder(const MaxCorrStrcut a, const MaxCorrStrcut b);//����
	void corrSort(MaxCorrStrcut* maxCorr, COMPARE_MODE sortMode, int size);

	static int findMaxCorr(float* corr,int len);
	static void bubbleSort(float arr[],int8_t wt[],int len);
	static int intlen(int number);

	static double aver(float* array,int len);
	static double var(float* array,int len);
	static double cov(float* array1,float* array2,int len);

	static void generateFileName(char* name,int buflen,char* fixedName,int index,char* postfix);
	static int hToD(char* str);
	static void byteToH(uint8_t i, char* str);
	static int charToNumD(char* str);
};

