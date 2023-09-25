#include "../../Inc/BASETOOL/BaseTools.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include<string.h>
using namespace std;

int BaseTools::hanmingWeight(uint8_t data)
{
	int n = 0;
	for (int j = 0; j < 8; j++)
		n += (data >> (7 - j)) & 0x1;
	return n;
}

int BaseTools::hanmingWeight(int8_t data)
{
	int n = 0;
	for (int j = 0; j < 8; j++)
		n += (data >> (7 - j)) & 0x1;
	return n;
}

int BaseTools::hanmingWeight(uint16_t data)
{
	int n = 0;
	for (int j = 0; j < 16; j++)
		n += (data >> (15 - j)) & 0x1;
	return n;
}

int BaseTools::hanmingWeight(uint32_t data)
{
	int n = 0;
	for (int j = 0; j < 32; j++)
		n += (data >> (31 - j)) & 0x1;
	return n;
}


int BaseTools::hanmingWeight(uint64_t data)
{
	int n = 0;
	for (int j = 0; j < 64; j++)
		n += (data >> (63 - j)) & 0x1;
	return n;
}

int BaseTools::hanmingWeight(const int32_t data) {
	int n = 0;
	for (int j = 0; j < 32; j++)
		n += (data >> (31 - j)) & 0x1;
	return n;
}

int BaseTools:: hanmingWeight(const int32_t data,int bias){
	
	int n = 0;
	for (int j = 0; j < bias; j++)
		n += (data >> (bias - 1 - j)) & 0x1;
	return n;
	
}

double BaseTools::correlate(double* src1, double* src2, int len)
{
	double sum_src1 = 0;
	double sum_src2 = 0;
	double ave_src1 = 0;
	double ave_src2 = 0;
	double deviation_src1 = 0;
	double deviation_src2 = 0;
	double cov_xy = 0;
	for (int i = 0; i < len; i++)
	{
		sum_src1 += src1[i];
		sum_src2 += src2[i];
	}
	ave_src1 = sum_src1 / len;
	ave_src2 = sum_src2 / len;
	// ����src1�ķ������src2�ķ���
	for (int i = 0; i < len; i++) {
		deviation_src2 += (src2[i] - ave_src2) * (src2[i] - ave_src2);
		deviation_src1 += (src1[i] - ave_src1) * (src1[i] - ave_src1);
		cov_xy += (src2[i] - ave_src2) * (src1[i] - ave_src1);
	}
	deviation_src1 = (std::sqrt(deviation_src1));
	deviation_src2 = (std::sqrt(deviation_src2));
	if (0 == deviation_src1 * deviation_src2)
	{
		return 0;
	}
	else
	{
		return cov_xy / (deviation_src1 * deviation_src2);
	}
}

double BaseTools::correlate(float* src1, float* src2, int len)
{
	double sum_src1 = 0;
	double sum_src2 = 0;
	double ave_src1 = 0;
	double ave_src2 = 0;
	double deviation_src1 = 0;
	double deviation_src2 = 0;
	double cov_xy = 0;
	for (int i = 0; i < len; i++)
	{
		sum_src1 += src1[i];
		sum_src2 += src2[i];
	}
	ave_src1 = sum_src1 / len;
	ave_src2 = sum_src2 / len;
	// ����src1�ķ������src2�ķ���
	for (int i = 0; i < len; i++) {
		deviation_src2 += (src2[i] - ave_src2) * (src2[i] - ave_src2);
		deviation_src1 += (src1[i] - ave_src1) * (src1[i] - ave_src1);
		cov_xy += (src2[i] - ave_src2) * (src1[i] - ave_src1);
	}
	deviation_src1 = (std::sqrt(deviation_src1));
	deviation_src2 = (std::sqrt(deviation_src2));
	if (0 == deviation_src1 * deviation_src2)
	{
		return 0;
	}
	else
	{
		return cov_xy / (deviation_src1 * deviation_src2);
	}
}

//bool BaseTools::descendingOrder(const MaxCorrStrcut a, const MaxCorrStrcut b)
//{
//	return a.corr > b.corr;//����
//}
//
//bool BaseTools::ascendingOrder(const MaxCorrStrcut a, const MaxCorrStrcut b)
//{
//	return a.corr < b.corr;//����
//}

void BaseTools::corrSort(MaxCorrStrcut* maxCorr, COMPARE_MODE sortMode, int size)
{
	if (sortMode == DESCENDING_SORT)
	{
		//����
		sort(maxCorr, maxCorr + size,descendingOrder);//����
	}
	else
	{
		//����
		sort(maxCorr, maxCorr + size,ascendingOrder);//����
	}
	
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double BaseTools::corr(uint8_t* hw_arrays, float* power, int len) {
	// for(int m=0;m<10;m++){
    //     printf("%d,",hw_arrays[m]);
    // }
    // printf("\n");
    // for(int n=0;n<10;n++){
    //     printf("%f,",power[n]);
    // }
    // printf("\n");
	double sum_hw = 0;
	double sum_trace = 0;
	double ave_hw = 0;
	double ave_trace = 0;
	double deviation_hw = 0;
	double deviation_trace = 0;
	double sqrt_deviation_hw = 0;
	double sqrt_deviation_trace = 0;
	double cov_xy = 0;
	for (int i = 0; i < len; i++)
	{
		// printf("%f,",(double)hw_arrays[i]);
		sum_hw += (double)hw_arrays[i];
		sum_trace += (double)power[i];
	}
	// printf("\n");
	// printf("%f,",sum_hw);
	ave_hw = sum_hw / len;
	// printf("%f,",ave_hw);
	ave_trace = sum_trace / len;
	// printf("%f,",sum_trace);
	// printf("%f,",ave_trace);
	// 计算hw的方差，计算trace的方差
	for (int i = 0; i < len; i++) {
		deviation_trace += ((double)power[i] - ave_trace) * ((double)power[i] - ave_trace);
		deviation_hw += ((double)hw_arrays[i] - ave_hw) * ((double)hw_arrays[i] - ave_hw);
		cov_xy += ((double)power[i] - ave_trace) * ((double)hw_arrays[i] - ave_hw);
	}
	sqrt_deviation_hw = std::sqrt(deviation_hw);
	sqrt_deviation_trace = std::sqrt(deviation_trace);
	if(sqrt_deviation_hw==0|sqrt_deviation_trace==0){
		return 0;
	}else{
		return cov_xy / (sqrt_deviation_hw * sqrt_deviation_trace);
	}
	//  printf("%lf\n",cov_xy / (sqrt_deviation_hw * sqrt_deviation_trace));
	
}

int BaseTools::findMaxCorr(float* corr,int len){
	float max=0.0;
	int index=0;
	for(int i=0;i<len;i++){
		if(corr[i]>max){
			max=corr[i];
			index=i;
		}

	}
	return index;
}


void BaseTools::bubbleSort(float arr[],int8_t wt[],int len){
    float temp=0;
    int temp_index=0;
    

    for(int i=0;i<len-1;i++){
        for(int j=0;j<len-1;j++){
            if(arr[j]<arr[j+1]){
                temp=arr[j];
                arr[j]=arr[j+1];
                arr[j+1]=temp;

                temp_index=wt[j];
                wt[j]=wt[j+1];
                wt[j+1]=temp_index;
            }
        }
    }
   
}

int BaseTools::intlen(int number){
	int i=1;
    while(1){
        number/=10;
        if(number){
            i++;
        }else{
            break;
        }
    }
    return i;
}

double BaseTools::diff(uint8_t* hw_arrays, float* power, int len, int thresholdL) {
	double hUp=0;
	int hUpNum=0;
	double hDown=0;
	int hDownNum=0;

	for(int i=0;i<len;i++){
		if(hw_arrays[i]>=thresholdL){
			hUp+=power[i];
			hUpNum++;
		}else {
			hDown+=power[i];
			hDownNum++;
		}
	}

	if(hUpNum) hUp=hUp/hUpNum;
		else return 0;
	if(hDownNum) hDown=hDown/hDownNum;
		else return 0;
		
	
	return hUp-hDown;
	

}



double BaseTools::ttest(float* sample1, float* sample2, int len1, int len2){
	double result = 0.0;

	result = (aver(sample1, len1)-aver(sample2, len2)) / pow((1/(float)len1 + 1/(float)len2), 0.5) / pow(((float)(len1-1)*var(sample1, len1)*var(sample1, len1) + (float)(len2-1)*var(sample2, len2)*var(sample2, len2)) / (float)(len1+len2-2), 0.5);

	return result;
}

double BaseTools::x2test(uint8_t* hw_arrays, float* power, int len){
	double result = 0.0;
	for(int i = 0; i< len;i++){
		result+=(power[i]- (float)hw_arrays[i])*(power[i]- (float)hw_arrays[i])/(float)hw_arrays[i];
	}
	return result;
}


double BaseTools::aver(float* array,int len){
	double sum=0.0;
	for(int i=0;i<len;i++){
		sum+=(double)array[i];
	}
	return sum/len;
}

double BaseTools::var(float* array,int len){
	double ave = aver(array,len);
	double res=0.0;
	for(int i=0;i<len;i++){
		res+=((double)array[i]-ave)*((double)array[i]-ave);
	}
	return res/len;
}

double BaseTools::cov(float* array1,float* array2,int len){
	double ave1=aver(array1,len);
	double ave2=aver(array2,len);
	double res=0.0;

	for(int i=0;i<len;i++){
		res+=((double)array1[i]-ave1)*((double)array2[i]-ave2);
	}

	return res/len;
}

int BaseTools::hdistance(int x, int y)
{
    int dist = 0;
    unsigned  val = x ^ y;
    
    // Count the number of bits set
    while (val != 0)
    {
        // A bit is set, so increment the count and clear the bit
        dist++;
        val &= val - 1;
    }
    
    // Return the number of differing bits
    return dist;
}

////////////////////////////////////////////////////////////////////////////////

void BaseTools::generateFileName(char* name,int buflen,char* fixedName,int index,char* postfix){
	#if 0
	memset(name,'\0',buflen);
	char strIndex[256];
	strcat(name,fixedName);
	itoa(index,strIndex,10);
	strcat(name,strIndex);
	strcat(name,postfix);
	#endif
}

void BaseTools::byteToH(uint8_t ui8, char* str){
	
	uint8_t i = ui8/16;
	uint8_t j = ui8%16;

	if (i == 0){
		str[0] = '0';
	}else if(i == 1){
		str[0] = '1';
	}else if(i == 2){
		str[0] = '2';
	}else if(i == 3){
		str[0] = '3';
	}else if(i == 4){
		str[0] = '4';
	}else if(i == 5){
		str[0] = '5';
	}else if(i == 6){
		str[0] = '6';
	}else if(i == 7){
		str[0] = '7';
	}else if(i == 8){
		str[0] = '8';
	}else if(i == 9){
		str[0] = '9';
	}else if(i == 10){
		str[0] = 'A';
	}else if(i == 11){
		str[0] = 'B';
	}else if(i == 12){
		str[0] = 'C';
	}else if(i == 13){
		str[0] = 'D';
	}else if(i == 14){
		str[0] = 'E';
	}else if(i == 15){
		str[0] = 'F';
	}

	if (j == 0){
		str[1] = '0';
	}else if(j == 1){
		str[1] = '1';
	}else if(j == 2){
		str[1] = '2';
	}else if(j == 3){
		str[1] = '3';
	}else if(j == 4){
		str[1] = '4';
	}else if(j == 5){
		str[1] = '5';
	}else if(j == 6){
		str[1] = '6';
	}else if(j == 7){
		str[1] = '7';
	}else if(j == 8){
		str[1] = '8';
	}else if(j == 9){
		str[1] = '9';
	}else if(j == 10){
		str[1] = 'A';
	}else if(j == 11){
		str[1] = 'B';
	}else if(j == 12){
		str[1] = 'C';
	}else if(j == 13){
		str[1] = 'D';
	}else if(j == 14){
		str[1] = 'E';
	}else if(j == 15){
		str[1] = 'F';
	}

}

int BaseTools::hToD(char* str){
	int re=0;

	if(str[0]=='0'){

	}else if(str[0]=='1'){
		re+=16;
	}else if(str[0]=='2'){
		re+=2*16;
	}else if(str[0]=='3'){
		re+=3*16;
	}else if(str[0]=='4'){
		re+=4*16;
	}else if(str[0]=='5'){
		re+=5*16;
	}else if(str[0]=='6'){
		re+=6*16;
	}else if(str[0]=='7'){
		re+=7*16;
	}else if(str[0]=='8'){
		re+=8*16;
	}else if(str[0]=='9'){
		re+=9*16;
	}else if(str[0]=='a'|str[0]=='A'){
		re+=10*16;
	}else if(str[0]=='b'|str[0]=='B'){
		re+=11*16;
	}else if(str[0]=='c'|str[0]=='C'){
		re+=12*16;
	}else if(str[0]=='d'|str[0]=='D'){
		re+=13*16;
	}else if(str[0]=='e'|str[0]=='E'){
		re+=14*16;
	}else if(str[0]=='f'|str[0]=='F'){
		re+=15*16;
	}

	if(str[1]=='0'){

	}else if(str[1]=='1'){
		re+=1;
	}else if(str[1]=='2'){
		re+=2;
	}else if(str[1]=='3'){
		re+=3;
	}else if(str[1]=='4'){
		re+=4;
	}else if(str[1]=='5'){
		re+=5;
	}else if(str[1]=='6'){
		re+=6;
	}else if(str[1]=='7'){
		re+=76;
	}else if(str[1]=='8'){
		re+=8;
	}else if(str[1]=='9'){
		re+=9;
	}else if(str[1]=='a'|str[1]=='A'){
		re+=10;
	}else if(str[1]=='b'|str[1]=='B'){
		re+=11;
	}else if(str[1]=='c'|str[1]=='C'){
		re+=12;
	}else if(str[1]=='d'|str[1]=='D'){
		re+=13;
	}else if(str[1]=='e'|str[1]=='E'){
		re+=14;
	}else if(str[1]=='f'|str[1]=='F'){
		re+=15;
	}

	return re;
}

int BaseTools::charToNumD(char* str){

	int re = 0;
	int num[1024]={0};
	int numLen = 0;
	int isPostive = 1;
	// int i = 0;
	while(str[numLen] != '\0'){
		if(str[numLen]=='-'){
			isPostive = 0;
		}else if(str[numLen]=='0'){
			num[numLen] = 0;
		}else if(str[numLen]=='1'){
			num[numLen] = 1;
		}else if(str[numLen]=='2'){
			num[numLen] = 2;
		}else if(str[numLen]=='3'){
			num[numLen] = 3;
		}else if(str[numLen]=='4'){
			num[numLen] = 4;
		}else if(str[numLen]=='5'){
			num[numLen] = 5;
		}else if(str[numLen]=='6'){
			num[numLen] = 6;
		}else if(str[numLen]=='7'){
			num[numLen] = 7;
		}else if(str[numLen]=='8'){
			num[numLen] = 8;
		}else if(str[numLen]=='9'){
			num[numLen] = 9;
		}
    // i++;
		numLen ++;
	}

  
  	int w=1;
	for(int i =numLen;i>= 0;i--){

		if(i == numLen){
		re += num[i];
		}else{
			re += num[i] * w;
		w*=10;
		}
    
	}

	if(isPostive){
		return re;
	}else{
		return -re;
	}
	

}
