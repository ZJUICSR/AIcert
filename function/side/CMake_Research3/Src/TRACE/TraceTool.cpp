#include "../../Inc/TRACE/TraceTool.h"
#include "../../Inc/BASETOOL/BaseTools.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <vector>
// #include "../../eigen/Eigen/Dense"
// #include "../../eigen/Eigen/Eigenvalues"
// using namespace Eigen;


Trs TraceTools::addTrace(Trs trs1, Trs trs2)
{
	Trs resultTrsData;
	resultTrsData.trsHead = trs1.trsHead;
	resultTrsData.trsData = new TrsData[resultTrsData.trsHead.NT];
	for (int i = 0; i < trs1.trsHead.NT; i++)
	{
		for (int j = 0; j < trs1.trsHead.DS; j++)
		{
			resultTrsData.trsData[i].data[j] = trs1.trsData[i].data[j] + trs2.trsData[i].data[j];
		}
		for (int j = 0; j < trs1.trsHead.NS; j++)
		{
			resultTrsData.trsData[i].samples[j] = (float)(trs1.trsData[i].samples[j] + trs2.trsData[i].samples[j]);
		}
	}
	return resultTrsData;
}

Trs TraceTools::subTrace(Trs trs1, Trs trs2)
{
	Trs resultTrsData;
	resultTrsData.trsHead = trs1.trsHead;
	resultTrsData.trsData = new TrsData[resultTrsData.trsHead.NT];
	for (int i = 0; i < trs1.trsHead.NT; i++)
	{
		for (int j = 0; j < trs1.trsHead.DS; j++)
		{
			resultTrsData.trsData[i].data[j] = trs1.trsData[i].data[j] - trs2.trsData[i].data[j];
		}
		for (int j = 0; j < trs1.trsHead.NS; j++)
		{
			resultTrsData.trsData[i].samples[j] = trs1.trsData[i].samples[j] - trs2.trsData[i].samples[j];
		}
	}
	return resultTrsData;
}

//平均一组Trace的功耗
TrsData TraceTools::meanTrace(Trs trs)
{
	float sum = 0;
	TrsData trsData;
	trsData.data = new uint8_t[trs.trsHead.DS];
	trsData.samples = new float[trs.trsHead.NS];
	double sumSample = 0;
	int sumData = 0;
	for (int j = 0; j < trs.trsHead.DS; j++)
	{
		for (int i = 0; i < trs.trsHead.NT; i++)
		{
			sumData += trs.trsData[i].data[j];
		}
		trsData.data[j] = sumData / trs.trsHead.NT;
	}
	for (int j = 0; j < trs.trsHead.NS; j++)
	{
		for (int i = 0; i < trs.trsHead.NT; i++)
		{
			sumSample += trs.trsData[i].samples[j];
		}
		trsData.samples[j] = sumSample / trs.trsHead.NT;
	}
	return trsData;
}

void TraceTools::meanTrace(TrsData* outTrsData, TrsHead trsHead, TrsData* inTrsData)
{
	outTrsData->samples = new float[trsHead.NS];
	for (int j = 0; j < trsHead.NS; j++)//曲线点数
	{
		double sumSample = 0;
		for (int i = 0; i < trsHead.NT; i++)//曲线量
		{
			sumSample += inTrsData[i].samples[j];
		}
		outTrsData->samples[j] = sumSample / trsHead.NT;
	}
}

void TraceTools::trs2txt(const char* trsFileName)
{
	Trace* trace = new Trace(trsFileName);
	// 打开文件
	ofstream outWave("./outtxt/wave.txt", ios::out | ios::trunc);
	ofstream outMessage("./outtxt/in.txt", ios::out | ios::trunc);
	ofstream outCipher("./outtxt/out.txt", ios::out | ios::trunc);
	if (!outWave.is_open() || !outMessage.is_open() || !outCipher.is_open())
	{
		cerr << "Failed to open the file!";
		exit(0);
	}
	//trace->readAllTrace(trsData);
	TrsData* trsData = new TrsData;
	while (trace->currentTrace < trace->trsHead.NT)
	{
		trace->readNext(trsData);
		for (int i = 0; i < trace->trsHead.NS; i++)
		{
			//outWave.write((char*)(trsData->samples+i),4);
			outWave << trsData->samples[i] << " ";
		}
		outWave << endl;
		// LAC 明文
		for (int i = 0; i < 648; i++)
		{
			//outMessage.write((char*)trsData->data+i,1);
			outMessage << trsData->data[i] << " ";
		}
		outMessage << endl;
		// LAC 密文
		for (int i = 0; i < 16; i++)
		{
			//outCipher.write((char*)trsData->data+648 + i,1);
			outCipher << trsData->data[i + 648] << " ";
		}
		outCipher << endl;
		cout << trace->currentTrace << "/" << trace->trsHead.NT << endl;
	}
	outCipher.close();
	outMessage.close();
	outWave.close();
}

void TraceTools::trs2txt(char* trsFileName, int txtMessageLength, int txtCipherLength)
{
	Trace* trace = new Trace(trsFileName);
	// 打开文件
	ofstream outWave("./outtxt/wave.txt", ios::out | ios::trunc);
	ofstream outMessage("./outtxt/in.txt", ios::out | ios::trunc);
	ofstream outCipher("./outtxt/out.txt", ios::out | ios::trunc);
	if (!outWave.is_open() || !outMessage.is_open() || !outCipher.is_open())
	{
		cerr << "Failed to open the file!";
		exit(0);
	}
	//trace->readAllTrace(trsData);
	TrsData* trsData = new TrsData;
	while (trace->currentTrace < trace->trsHead.NT)
	{
		trace->readNext(trsData);
		for (int i = 0; i < trace->trsHead.NS; i++)
		{
			//outwave.write((char*)(trsData->samples+i*4),4);
			outWave << trsData->samples[i] << " ";
		}
		outWave << endl;
		// LAC 明文
		for (int i = 0; i < txtMessageLength; i++)
		{
			//outwave.write((char*)(trsData->samples+i*4),4);
			outMessage << trsData->data[i] << " ";
		}
		outMessage << endl;
		// LAC 密文
		for (int i = 0; i < txtCipherLength; i++)
		{
			//outwave.write((char*)(trsData->samples+i*4),4);
			outCipher << trsData->data[i + txtMessageLength] << " ";
		}
		outCipher << endl;
		cout << trace->currentTrace << "/" << trace->trsHead.NT << endl;
	}
	outCipher.close();
	outMessage.close();
	outWave.close();
}

//void TraceTools::trsD_input(TrsData trsdata, int bias) {
//	//trsdata.
//}


void TraceTools::pca(char* fileName, char *outfileName){

	#if 0

  ofstream *outfile = new ofstream();
  outfile->open(outfileName, ios::out | ios::binary | ios::trunc);
  Trace trace(fileName);
  TrsData trsData;
  TrsHead trsHead; //= trace.trsHead;
//   Trace::writeHead(outfile, trace.trsHead);
	// trsHead.NT = trace.trsHead.NT;
    // trsHead.NS = trace.trsHead.NS;
    // trsHead.DS = 0;
    // trsHead.YS = 1;
    // trsHead.SC = 0x14;//float存储类型
    // trsHead.GT_length = 0;
    // trsHead.DC_length = 0;
    // trsHead.XL_length = 0;
    // trsHead.YL_length = 0;
    // trsHead.TS = 0;
  Trace::writeHead(outfile, trace.trsHead);

  TrsData reTrsData;
  reTrsData.samples = (float *)malloc(trace.trsHead.NS * sizeof(float));
  std::memset(reTrsData.samples, 0.0, trace.trsHead.NS * sizeof(float));
  reTrsData.data = (uint8_t *)malloc(trace.trsHead.DS * sizeof(uint8_t));
  std::memset(reTrsData.data, 0, trace.trsHead.DS * sizeof(uint8_t));
  reTrsData.TSData = (uint8_t *)malloc(trace.trsHead.TS * sizeof(uint8_t));
  std::memset(reTrsData.TSData, 0, trace.trsHead.TS * sizeof(uint8_t));

  MatrixXd X(trace.trsHead.NS, trace.trsHead.NT), C(trace.trsHead.NT, trace.trsHead.NT);
  //定义特征向量矩阵vec，特征值val
  MatrixXd vec, val;
  //读取数据并放入数据矩阵X(m,n)
  for (int i = 0; i < trace.trsHead.NT; ++i)
  {
    trace.readIndexTrace(&trsData, i);
	memcpy(reTrsData.data,trsData.data,trace.trsHead.DS * sizeof(uint8_t));
	memcpy(reTrsData.TSData,trsData.TSData,trace.trsHead.TS * sizeof(uint8_t));
    for (int j = 0; j < trace.trsHead.NS; ++j)
    {
      X(j, i) = trsData.samples[j];
    }
  }
  //完成矢量量化
  //计算每一维度均值
  //零均值化
  MatrixXd meanval = X.colwise().mean();
  RowVectorXd meanvecRow = meanval;
  //样本均值化为0
  X.rowwise() -= meanvecRow;

  //计算协方差
  C = X.adjoint() * X;
  C = C.array() / (X.rows() - 1);
  //计算特征值和特征向量
  SelfAdjointEigenSolver<MatrixXd> eig(C);
  vec = eig.eigenvectors();
  val = eig.eigenvalues();
  //计算损失率，确定降低维数（原本的n维向量即降低到dim维向量）
  int dim;
  double sum = 0;
  for (int i = val.rows() - 1; i >= 0; --i)
  {
    sum += val(i, 0);
    dim = i;
    if (sum / val.sum() >= 0.95)
      break;
  }
  int k = val.rows() - dim;

  //定义投影矩阵V
  MatrixXd V(trace.trsHead.NT, k);
  //投影矩阵的值为特征向量矩阵的前k维
  V = vec.rightCols(k);

  //计算结果:把原始数据进行投影，每个数据点m维向量投影成k维向量,res为压缩后的数据
  MatrixXd res = X * V;

  //还原的中心化矩阵Y
  MatrixXd Y(trace.trsHead.NS, trace.trsHead.NT);
  Y = res * V.transpose();

  //样本每个维度都加上均值就是解压缩样本
  Y.rowwise() += meanvecRow;

  
  for (int i = 0; i < trace.trsHead.NT; ++i)
  {
    for (int j = 0; j < trace.trsHead.NS; ++j)
    {
      reTrsData.samples[j] = Y(j, i);
      // printf("%f,", reTrsData.samples[j]);
    }
    // cout << endl;
    Trace::writeNext(outfile, &reTrsData, trace.trsHead);
	
  }




  outfile->close();
  free(outfile);
  free(reTrsData.samples);
  free(trsData.samples);
  free(trsData.data);
  free(trsData.TSData);

  #endif


}



void TraceTools::cor(TrsData *baseTrsData, TrsData *srcTrsData, int samplePointNum, int length, int immp, TrsHead trsHead, const char *trsfileout){
	//    ofstream txtfile(txtfileout);
    TrsData trsData_result;
    trsData_result.samples = (float*)malloc((samplePointNum - length)*sizeof(float));
    for (int i = 0; i < samplePointNum - length; i++)
    {
        trsData_result.samples[i] = BaseTools::correlate(baseTrsData->samples, srcTrsData->samples + i + immp, length);
//        txtfile << trsData_result.samples[i] <<' ';
    }

    TrsHead trsHead_result = trsHead;
    trsHead_result.NT = 1;
    trsHead_result.NS = samplePointNum - length;
    trsHead_result.DS = 0;
    trsHead_result.YS = 1;
    trsHead_result.SC = 0x14;
    trsHead_result.GT_length = 0;
    trsHead_result.DC_length = 0;
    trsHead_result.XL_length = 0;
    trsHead_result.YL_length = 0;
    trsHead_result.TS = 0;
    Trace trace;

    trace.createTrace(trsfileout, &trsHead_result, &trsData_result);
//    txtfile.close()


	free(trsData_result.samples);
}

