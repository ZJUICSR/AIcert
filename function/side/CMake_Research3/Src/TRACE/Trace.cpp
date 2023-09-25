#include "../../Inc/TRACE/Trace.h"
#include <cstring>
#include <iostream>

#define BigtoLittle16(A) ((((uint16_t)(A) & 0xff00) >> 8) | (((uint16_t)(A) & 0x00ff) << 8))
#define BigtoLittle32(A) ((((uint32_t)(A) & 0xff000000) >> 24) | (((uint32_t)(A) & 0x00ff0000) >> 8) | (((uint32_t)(A) & 0x0000ff00) << 8) | (((uint32_t)(A) & 0x000000ff) << 24))


Trace::Trace()
{
	this->currentTrace = 0;//初始化
	this->trsHead.TS = 0;//初始化
}

Trace::Trace(const char* file) 
{
	// 打开文件
	this->infile.open(file, ios::in | ios::binary);
	if (!this->infile)
	{
		cerr << "Failed to open the file!";
		exit(0);
	}
	this->currentTrace = 0;//初始化
	this->trsHead.TS = 0;//初始化
	this->readHeard(file);//打开文件并读取头文件
}

Trace::Trace(const string file)
{
	// 打开文件
	this->infile.open(file, ios::in | ios::binary);
	if (!this->infile)
	{
		cerr << "Failed to open the file!";
		exit(0);
	}
	this->currentTrace = 0;//初始化
	this->trsHead.TS = 0;//初始化
	this->readHeard(file.data());//打开文件并读取头文件
}

Trace::Trace(const char* file, int inDataLen, int outDataLen)//默认读取模式为单条功耗
{
	// 打开文件
	this->infile.open(file, ios::in | ios::binary);
	if (!this->infile)
	{
		cerr << "Failed to open the file!";
		exit(0);
	}
	this->currentTrace = 0;
	this->outDataLen = outDataLen;
	this->inDataLen = inDataLen;
	this->trsHead.TS = 0;
	this->readHeard(file);
}

void Trace::readHeard(ifstream* inFile, TrsHead* trsHead)
{
	if (!inFile->is_open())
	{
		cerr << "Failed to open the file!";
		exit(0);
	}
	// 读取头信息
	//uint8_t data[4];
	uint8_t tag = 0;
	uint8_t length = 0;
	inFile->read((char*)&tag, 1);//读取第一个头部信息
	//力科示波器采集的trs头部信息顺序：NT NS SC DS TS GT NS XL XS YS
	//参考示例的头部信息顺序：NT NS SC DS GT
	while (tag != 0x5F)//头部信息TB
	{
		switch (tag)
		{
		case 0x41: //NT:Number of traces  ->trace条数
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->NT, 4);
			//cout << "NT=" << trsHead->NT << endl;
			break;
		case 0x42: //NS:Number of samples per trace  ->每条trace的总点数
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->NS, 4);
			//cout << "NS=" << trsHead->NS << endl;
			break;
		case 0x43: //SC:Sample coding (e.g. type and length in bytes of each sample)  ->功耗的存储类型：float/int/short/byte
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->SC, length);
			//printf("SC=%x\n", trsHead->SC);
			//std::cout << "SC=" << std::hex << trsHead->SC << endl;
			break;
		case 0x44: //DS:Length of cryptographic data included in trace	->加/解密的明密文总数据个数，即：len(明文)+len(密文)
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->DS, length);//length=2
			//cout << "DS=" << trsHead->DS << endl;
			break;
		case 0x45: //TS:Title space reserved per trace ->每条功耗的title
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->TS, length);//length =1 
			//cout << "TS=" << trsHead->TS << endl;
			//printf("TS=%d\n", trsHead->TS);
			break;
		case 0x46: //GT:Global trace title				->这组trace的title
			inFile->read((char*)&trsHead->GT_length, 1);
			trsHead->GT = new uint8_t[trsHead->GT_length];
			inFile->read((char*)trsHead->GT, trsHead->GT_length);
			//cout << "GT=" << trsHead->GT << endl;
			//printf("GT=");
			//for (int k = 0; k < trsHead->GT_length; k++)
			//{
			//	printf("%x", trsHead->GT[k]);
			//}
			//printf("\n");
			break;
		case 0x47: //DC:Description   ->没有该tag
			inFile->read((char*)&trsHead->DC_length, 1);
			trsHead->DC = new uint8_t[trsHead->DC_length];
			inFile->read((char*)trsHead->DC, trsHead->DC_length);
			//cout << "DC=" << trsHead->DC << endl;
			break;
		case 0x48: //XO:Offset in X-axis for trace representation	->没有该tag
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->XO, length);
			//cout << "XO=" << trsHead->XO << endl;
			break;
		case 0x49: //XL:Label of X-axis   时间的单位
			inFile->read((char*)&trsHead->XL_length, 1);
			trsHead->XL = new uint8_t[trsHead->XL_length];
			inFile->read((char*)trsHead->XL, trsHead->XL_length);
			//cout << "XL=" << trsHead->XL << endl;
			break;
		case 0x4A: //YL:Label of Y-axis   功耗的单位
			inFile->read((char*)&trsHead->YL_length, 1);
			trsHead->YL = new uint8_t[trsHead->YL_length];
			inFile->read((char*)trsHead->YL, trsHead->YL_length);
			//cout << "YL=" << trsHead->YL << endl;
			break;
		case 0x4B: //XS:Scale value for X-axis 时间缩放率
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->XS, length);
			//cout << "XS=" << trsHead->XS << endl;
			break;
		case 0x4C: //YS:Scale value for Y-axis 功耗缩放率
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->YS, length);
			//cout << "YS=" << trsHead->YS << endl;
			break;
		case 0x4D: //TO:Trace offset for displaying trace numbers
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->TO, length);
			//cout << "TO=" << trsHead->TO << endl;
			break;
		case 0x4E: //LS:Logarithmic scale
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->LS, length);
			//cout << "LS=" << trsHead->LS << endl;
			break;
		case 0x55: //RG:Range of the scope used to perform acquisition
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->RG, length);
			//cout << "RG=" << trsHead->RG << endl;
			break;
		case 0x56: //CL:Coupling of the scope used to perform acquisition
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->CL, length);
			//cout << "CL=" << trsHead->CL << endl;
			break;
		case 0x57: //OS:Offset of the scope used to perform acquisition
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->OS, length);
			//cout << "OS=" << trsHead->OS << endl;
			break;
		case 0x58: //II:Input impedance of the scope used to perform acquisition
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->II, length);
			//cout << "II=" << trsHead->II << endl;
			break;
		case 0x59: //AI:Device ID of the scope used to perform acquisition
			inFile->read((char*)&length, 1);
			trsHead->AI = new uint8_t[length];
			inFile->read((char*)trsHead->AI, length);
			//cout << "AI=" << trsHead->AI << endl;
			break;
		case 0x5A: //FT:The type of filter used during acquisition
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->FT, length);
			//cout << "FT=" << trsHead->FT << endl;
			break;
		case 0x5B: //FF:Frequency of the filter used during acquisition
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->FF, length);
			//cout << "FF=" << trsHead->FF << endl;
			break;
		case 0x5C: //FR:Range of the filter used during acquisition
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->FR, length);
			//cout << "FR=" << trsHead->FR << endl;
			break;
		case 0x60: //EU:External clock used
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->EU, length);
			//cout << "EU=" << trsHead->EU << endl;
			break;
		case 0x61: //ET:External clock used
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->ET, length);
			//cout << "ET=" << trsHead->ET << endl;
			break;
		case 0x62: //EM:External clock multiplier
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->EM, length);
			//cout << "EM=" << trsHead->EM << endl;
			break;
		case 0x63: //EP:External clock phase shift
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->EP, length);
			//cout << "EP=" << trsHead->EP << endl;
			break;
		case 0x64: //ER:External clock resampler mask
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->ER, length);
			//cout << "ER=" << trsHead->ER << endl;
			break;
		case 0x65: //RE:External clock resampler enabled
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->RE, length);
			//cout << "RE=" << trsHead->RE << endl;
			break;
		case 0x66: //EF:External clock frequency
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->EF, length);
			//cout << "EF=" << trsHead->EF << endl;
			break;
		case 0x67: //EB:External clock time base
			inFile->read((char*)&length, 1);
			inFile->read((char*)&trsHead->EB, length);
			//cout << "EB=" << trsHead.EB << endl;
			break;
		default:
			// cout << "default ";
			break;
		}
		// cout << endl;
		inFile->read((char*)&tag, 1);//读取下一个标签信息
	}
	//printf("this->DS = %d\n", trsHead.DS);
	inFile->read((char*)&length, 1);//读取一个没用的字符：00,因为头部信息以0x5F00为结束标志，目前使用0x5F为判断依据，所以还需再读取一个0x00
}

void Trace::readHeard(const char* file)
{
	// 读取头信息
	//uint8_t data[4];
	uint8_t tag = 0;
	uint8_t length = 0;
	this->infile.read((char*)&tag,1);//读取第一个头部信息
	//力科示波器采集的trs头部信息顺序：NT NS SC DS TS GT NS XL XS YS
	//参考示例的头部信息顺序：NT NS SC DS GT
	while (tag != 0x5F)//头部信息TB
	{
		switch (tag)
		{
		case 0x41: //NT:Number of traces  ->trace条数
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.NT,4);
			 //cout << "NT=" << this->trsHead.NT << endl;
			break;
		case 0x42: //NS:Number of samples per trace  ->每条trace的总点数
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.NS, 4);
			//cout << "NS=" << this->trsHead.NS << endl;
			break;
		case 0x43: //SC:Sample coding (e.g. type and length in bytes of each sample)  ->功耗的存储类型：float/int/short/byte
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.SC, length);
			//printf("SC=%x\n", this->trsHead.SC);
			//std::cout << "SC=" << std::hex << this->trsHead.SC << endl;
			break;
		case 0x44: //DS:Length of cryptographic data included in trace	->加/解密的明密文总数据个数，即：len(明文)+len(密文)
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.DS, length);//length=2
			//cout << "DS=" << this->trsHead.DS << endl;
			break;
		case 0x45: //TS:Title space reserved per trace ->每条功耗的title
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.TS, length);//length =1 
			//cout << "TS=" << this->trsHead.TS << endl;
			//printf("TS=%d\n", this->trsHead.TS);
			break;
		case 0x46: //GT:Global trace title				->这组trace的title
			this->infile.read((char*)&this->trsHead.GT_length, 1);
			this->trsHead.GT = new uint8_t[this->trsHead.GT_length];
			this->infile.read((char*)this->trsHead.GT, this->trsHead.GT_length);
			//cout << "GT=" << this->trsHead.GT << endl;
			//printf("GT=");
			//for (int k = 0; k < this->trsHead.GT_length; k++)
			//{
			//	printf("%x", this->trsHead.GT[k]);
			//}
			//printf("\n");
			break;
		case 0x47: //DC:Description   ->没有该tag
			this->infile.read((char*)&this->trsHead.DC_length, 1);
			this->trsHead.DC = new uint8_t[this->trsHead.DC_length];
			this->infile.read((char*)this->trsHead.DC, this->trsHead.DC_length);
			//cout << "DC=" << this->trsHead.DC << endl;
			break;
		case 0x48: //XO:Offset in X-axis for trace representation	->没有该tag
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.XO, length);
			//cout << "XO=" << this->trsHead.XO << endl;
			break;
		case 0x49: //XL:Label of X-axis   时间的单位
			this->infile.read((char*)&this->trsHead.XL_length, 1);
			this->trsHead.XL = new uint8_t[this->trsHead.XL_length];
			this->infile.read((char*)this->trsHead.XL, this->trsHead.XL_length);
			//cout << "XL=" << this->trsHead.XL << endl;
			break;
		case 0x4A: //YL:Label of Y-axis   功耗的单位
			this->infile.read((char*)&this->trsHead.YL_length, 1);
			this->trsHead.YL = new uint8_t[this->trsHead.YL_length];
			this->infile.read((char*)this->trsHead.YL, this->trsHead.YL_length);
			//cout << "YL=" << this->trsHead.YL << endl;
			break;
		case 0x4B: //XS:Scale value for X-axis 时间缩放率
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.XS, length);
			 //cout << "XS=" << this->trsHead.XS << endl;
			break;
		case 0x4C: //YS:Scale value for Y-axis 功耗缩放率
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.YS, length);
			//cout << "YS=" << this->trsHead.YS << endl;
			break;
		case 0x4D: //TO:Trace offset for displaying trace numbers
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.TO, length);
			 //cout << "TO=" << this->trsHead.TO << endl;
			break;
		case 0x4E: //LS:Logarithmic scale
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.LS, length);
			//cout << "LS=" << this->trsHead.LS << endl;
			break;
		case 0x55: //RG:Range of the scope used to perform acquisition
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.RG, length);
			 //cout << "RG=" << this->trsHead.RG << endl;
			break;
		case 0x56: //CL:Coupling of the scope used to perform acquisition
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.CL, length);
			//cout << "CL=" << this->trsHead.CL << endl;
			break;
		case 0x57: //OS:Offset of the scope used to perform acquisition
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.OS, length);
			//cout << "OS=" << this->trsHead.OS << endl;
			break;
		case 0x58: //II:Input impedance of the scope used to perform acquisition
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.II, length);
			//cout << "II=" << this->trsHead.II << endl;
			break;
		case 0x59: //AI:Device ID of the scope used to perform acquisition
			this->infile.read((char*)&length, 1);
			this->trsHead.AI = new uint8_t[length];
			this->infile.read((char*)this->trsHead.AI, length);
			//cout << "AI=" << this->trsHead.AI << endl;
			break;
		case 0x5A: //FT:The type of filter used during acquisition
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.FT, length);
			//cout << "FT=" << this->trsHead.FT << endl;
			break;
		case 0x5B: //FF:Frequency of the filter used during acquisition
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.FF, length);
			//cout << "FF=" << this->trsHead.FF << endl;
			break;
		case 0x5C: //FR:Range of the filter used during acquisition
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.FR, length);
			//cout << "FR=" << this->trsHead.FR << endl;
			break;
		case 0x60: //EU:External clock used
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.EU, length);
			//cout << "EU=" << this->trsHead.EU << endl;
			break;
		case 0x61: //ET:External clock used
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.ET, length);
			//cout << "ET=" << this->trsHead.ET << endl;
			break;
		case 0x62: //EM:External clock multiplier
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.EM, length);
			//cout << "EM=" << this->trsHead.EM << endl;
			break;
		case 0x63: //EP:External clock phase shift
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.EP, length);
			//cout << "EP=" << this->trsHead.EP << endl;
			break;
		case 0x64: //ER:External clock resampler mask
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.ER, length);
			//cout << "ER=" << this->trsHead.ER << endl;
			break;
		case 0x65: //RE:External clock resampler enabled
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.RE, length);
			//cout << "RE=" << this->trsHead.RE << endl;
			break;
		case 0x66: //EF:External clock frequency
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.EF, length);
			//cout << "EF=" << this->trsHead.EF << endl;
			break;
		case 0x67: //EB:External clock time base
			this->infile.read((char*)&length, 1);
			this->infile.read((char*)&this->trsHead.EB, length);
			//cout << "EB=" << this->trsHead.EB << endl;
			break;
		default:
			// cout << "default ";
			break;
		}
		// cout << endl;
		this->infile.read((char*)&tag, 1);//读取下一个标签信息
	}
	//printf("this->DS = %d\n", this->trsHead.DS);
	this->infile.read((char*)&length, 1);//读取一个没用的字符：00,因为头部信息以0x5F00为结束标志，目前使用0x5F为判断依据，所以还需再读取一个0x00
}

//注意，有时trs文件的点数过多时，会内存溢出，导致无法全部读取
void Trace::readAllTrace(TrsData* trsData)
{
	for (int i = 0; i < this->trsHead.NT; i++)
	{
		if (0 != this->trsHead.TS)//如果每条功耗有title的话，读取出来
		{
			trsData[i].TSData = new uint8_t[this->trsHead.TS];
			this->infile.read((char*)trsData[i].TSData, this->trsHead.TS);
		}
		if (0 != this->trsHead.DS)//加密/解密前后的明密文数据不为空时
		{
			//申请明密文data存储空间
			trsData[i].data = new uint8_t[this->trsHead.DS];
			infile.read((char*)trsData[i].data, this->trsHead.DS);
		}
		if (0 != this->trsHead.NS)
		{
			if (this->trsHead.SC & 0x10)//SC的第5bit为1时，功耗数据存储类型为float
			{
				trsData[i].samples = new float[this->trsHead.NS];
				for (int j = 0; j < this->trsHead.NS; j++)
				{
					infile.read((char*)(trsData[i].samples + j), 4);//sizeof(float) = 4
					trsData->samples[j] = trsData->samples[j] * trsHead.YS;		//统一赋值给samples
				}
			}
			else//SC的第5bit为0时，功耗数据存储类型为int
			{
				trsData[i].samples = new float[this->trsHead.NS];
				switch (this->trsHead.SC & 0x07)
				{
				case 1:
					for (int j = 0; j < this->trsHead.NS; j++)
					{
						uint8_t data = 0;
						infile.read((char*)&data, 1);
						trsData[i].samples[j] = data * trsHead.YS;		//统一赋值给samples
					}
					break;
				case 2:
					for (int j = 0; j < this->trsHead.NS; j++)
					{
						short data = 0;
						infile.read((char*)&data, 2);
						trsData[i].samples[j] = data * trsHead.YS;		//统一赋值给samples
					}
					break;
				case 4:
					for (int j = 0; j < this->trsHead.NS; j++)
					{
						int data = 0;
						infile.read((char*)&data, 4);
						trsData[i].samples[j] = data * trsHead.YS;		//统一赋值给samples
					}
					break;
				default:
					break;
				}
			}
		}
	}
	
}

void Trace::readNext(TrsData* trsData)
{
	// 读取下一条数据
	if (currentTrace < this->trsHead.NT)
	{
		if (0 != this->trsHead.TS)//如果每条功耗有title的话，读取出来
		{
			if (trsData->TSData)
			{
				delete trsData->TSData;
				trsData->TSData = nullptr;
			}
			trsData->TSData = new uint8_t[this->trsHead.TS];
			infile.read((char*)trsData->TSData, this->trsHead.TS);
		}
		if (0 != this->trsHead.DS)//加密/解密前后的明密文数据不为空时
		{
			if (trsData->data)
			{
				delete trsData->data;
				trsData->data = nullptr;
			}
			//申请明密文data存储空间
			trsData->data = new uint8_t[this->trsHead.DS];
			infile.read((char*)trsData->data, this->trsHead.DS);
		}
		if (0 != this->trsHead.NS)
		{
			if (trsData->samples)
			{
				delete trsData->samples;
				trsData->samples = nullptr;
			}
			trsData->samples = new float[this->trsHead.NS];
			switch (SC_TYPE(this->trsHead.SC))
			{
			case FLOAT:
				for (int j = 0; j < this->trsHead.NS; j++)
				{
					infile.read((char*)(trsData->samples + j), 4);				//sizeof(float) = 4
					trsData->samples[j] = trsData->samples[j] * trsHead.YS;		//统一赋值给samples
				}
				break;
			case INT:
				for (int j = 0; j < this->trsHead.NS; j++)
				{
					int data = 0;
					infile.read((char*)&data, 4);
					trsData->samples[j] = data * trsHead.YS;		//统一赋值给samples
				}
				break;
			case SHORT:
				for (int j = 0; j < this->trsHead.NS; j++)
				{
					short data = 0;
					infile.read((char*)&data, 2);
					trsData->samples[j] = data * trsHead.YS;		//统一赋值给samples
				}
				break;
			case BYTE:
				for (int j = 0; j < this->trsHead.NS; j++)
				{
					char data = 0;
					infile.read((char*)&data, 1);
					trsData->samples[j] = data * trsHead.YS;		//统一赋值给samples
				}
				break;
			default:
				break;
			}
		}
	}
	currentTrace ++;
}

void Trace::readNext(ifstream* inFile, TrsData* trsData, const TrsHead trsHead, int* currentTrace)
{
	if (!inFile->is_open())
	{
		cerr << "Failed to open the file!";
		exit(0);
	}
	// 读取下一条数据
	if (*currentTrace < trsHead.NT)
	{
		if (0 != trsHead.TS)//如果每条功耗有title的话，读取出来
		{
			if (trsData->TSData)
			{
				delete trsData->TSData;
				trsData->TSData = nullptr;
			}
			trsData->TSData = new uint8_t[trsHead.TS];
			inFile->read((char*)trsData->TSData, trsHead.TS);
		}
		if (0 != trsHead.DS)//加密/解密前后的明密文数据不为空时
		{
			if (trsData->data)
			{
				delete trsData->data;
				trsData->data = nullptr;
			}
			//申请明密文data存储空间
			trsData->data = new uint8_t[trsHead.DS];
			inFile->read((char*)trsData->data, trsHead.DS);
		}
		if (0 != trsHead.NS)
		{
			if (trsData->samples)
			{
				delete trsData->samples;
				trsData->samples = nullptr;
			}
			trsData->samples = new float[trsHead.NS];
			switch (SC_TYPE(trsHead.SC))
			{
			case FLOAT:
				for (int j = 0; j < trsHead.NS; j++)
				{
					inFile->read((char*)(trsData->samples + j), 4);				//sizeof(float) = 4
					trsData->samples[j] = trsData->samples[j] * trsHead.YS;		//统一赋值给samples
				}
				break;
			case INT:
				for (int j = 0; j < trsHead.NS; j++)
				{
					int data = 0;
					inFile->read((char*)&data, 4);
					trsData->samples[j] = data * trsHead.YS;		//统一赋值给samples
				}
				break;
			case SHORT:
				for (int j = 0; j < trsHead.NS; j++)
				{
					short data = 0;
					inFile->read((char*)&data, 2);
					trsData->samples[j] = data * trsHead.YS;		//统一赋值给samples
				}
				break;
			case BYTE:
				for (int j = 0; j < trsHead.NS; j++)
				{
					char data = 0;
					inFile->read((char*)&data, 1);
					trsData->samples[j] = data * trsHead.YS;		//统一赋值给samples
				}
				break;
			default:
				break;
			}
		}
	}
	*currentTrace++;
}

void Trace::readIndexTrace(ifstream* inFile, TrsData* trsData, const TrsHead trsHead, int index, int* currentTrace)
{
	if (!inFile->is_open())
	{
		cerr << "Failed to open the file!";
		exit(0);
	}
	//下标越界检查
	if (inFile->eof() || index < 0)
	{
		cerr << "index trace out of trace!";
		exit(0);
	}
	int sampleSize = 0;
	switch (SC_TYPE(trsHead.SC))
	{
	case FLOAT:
		sampleSize = sizeof(float);
		break;
	case INT:
		sampleSize = sizeof(int);
		break;
	case SHORT:
		sampleSize = sizeof(short);
		break;
	case BYTE:
		sampleSize = sizeof(char);
		break;
	default:
		break;
	}
	int offset = (index - *currentTrace) * (trsHead.TS + trsHead.DS + sampleSize * trsHead.NS);
	//文件指针偏移设置
	inFile->seekg(offset, ios::cur);
	//下标越界检查
	if (inFile->eof())
	{
		//改回原来的位置
		inFile->seekg(-offset, ios::cur);
		cerr << "index trace out of trace!";
		exit(0);
	}
	//设置当前曲线下标
	*currentTrace = index;
	readNext(inFile, trsData, trsHead, currentTrace);
}

void Trace::writeNext(ofstream* outFile, TrsData* trsData, const TrsHead trsHead)
{
	if (0 != trsHead.TS)
	{
		outFile->write((char*)trsData->TSData, trsHead.TS);
	}
	if (0 != trsHead.DS)
	{
		outFile->write((char*)trsData->data, trsHead.DS);
	}
	switch (SC_TYPE(trsHead.SC))
	{
	case FLOAT:
		for (int j = 0; j < trsHead.NS; j++)
		{
			float value_float = 0;
			value_float = (float)(trsData->samples[j] / trsHead.YS);
			outFile->write((char*)&value_float, sizeof(float));
			//cout << "write：FLOAT=" << value_float << endl;
		}
		break;
	case INT:
		for (int j = 0; j < trsHead.NS; j++)
		{
			int value_int = 0;
			value_int = (int)(trsData->samples[j] / trsHead.YS);
			outFile->write((char*)&value_int, sizeof(int));
			//cout << "write：INT=" << value_int << endl;
		}
		break;
	case SHORT:
		for (int j = 0; j < trsHead.NS; j++)
		{
			short value_short = 0;
			value_short = (short)(trsData->samples[j] / trsHead.YS);
			outFile->write((char*)&value_short, sizeof(short));
			//cout << "write：SHORT=" << value_short << endl;
		}
		break;
	case BYTE:
		for (int j = 0; j < trsHead.NS; j++)
		{
			char value_byte = 0;
			value_byte = (char)(trsData->samples[j] / trsHead.YS);
			outFile->write((char*)&value_byte, sizeof(char));
			//cout << "write：BYTE=" << value_byte << endl;
		}
		break;
	default:
		break;
	}
}

void Trace::writeHead(ofstream* outFile, const TrsHead trsHead)
{
	if (!outFile->is_open())
	{
		cerr << "Failed to open the file!";
		exit(0);
	}
	//outFile->seekp(ios::beg);
	/***************** trsfile header ************************/
	char length = 4;
	char tag = 0x41;
	//***NT:Number of traces  ->trace条数
	if (0 != trsHead.NT)
	{
		tag = 0x41;
		length = 4;
		outFile->write(&tag, 1);
		outFile->write(&length, 1);
		outFile->write((char*)&trsHead.NT, length);
	}
	//***NS:Number of samples per trace  ->每条trace的总点数
	if (0 != trsHead.NS)
	{
		tag = 0x42;
		length = 4;
		outFile->write(&tag, 1);
		outFile->write(&length, 1);
		outFile->write((char*)&trsHead.NS, length);
	}
	//***SC:Sample coding (e.g. type and length in bytes of each sample)  ->功耗的存储类型：float/int/short/byte
	if (0 != trsHead.SC)
	{
		tag = 0x43;
		length = 1;
		outFile->write(&tag, 1);
		outFile->write(&length, 1);
		outFile->write((char*)&trsHead.SC, length);
	}
	//***DS:Length of cryptographic data included in trace	->加/解密的明密文总数据个数，即：len(明文)+len(密文)
	if (0 != trsHead.DS)
	{
		tag = 0x44;
		length = 2;
		outFile->write(&tag, 1);
		outFile->write(&length, 1);
		outFile->write((char*)&trsHead.DS, length);
	}
	//***TS:Title space reserved per trace ->每条功耗的title
	if (0 != trsHead.TS)
	{
		tag = 0x45;
		length = 1;
		outFile->write(&tag, 1);
		outFile->write(&length, 1);
		outFile->write((char*)&trsHead.TS, length);
	}
	//***GT:Global trace title				->这组trace的title
	if (0 != trsHead.GT_length)
	{
		tag = 0x46;
		outFile->write(&tag, 1);
		outFile->write((char*)&trsHead.GT_length, 1);
		outFile->write((char*)trsHead.GT, trsHead.GT_length);
		//cout << "trsHead.GT=" << trsHead.GT << endl;
	}
	//***DC:Description   ->没有该tag
	if (0 != trsHead.DC_length)
	{
		tag = 0x47;
		outFile->write(&tag, 1);
		outFile->write((char*)&trsHead.DC_length, 1);
		outFile->write((char*)trsHead.DC, trsHead.DC_length);
		//cout << "trsHead.DC=" << trsHead.DC << endl;
	}
	//***XO:Offset in X-axis for trace representation	->没有该tag
	if (0 != trsHead.XO)
	{
		tag = 0x48;
		length = 4;
		outFile->write(&tag, 1);
		outFile->write(&length, 1);
		outFile->write((char*)&trsHead.XO, length);
	}
	//***XL:Label of X-axis   时间的单位
	if (0 != trsHead.XL_length)
	{
		tag = 0x49;
		outFile->write(&tag, 1);
		outFile->write((char*)&trsHead.XL_length, 1);
		outFile->write((char*)trsHead.XL, trsHead.XL_length);
		//cout << "trsHead.XL=" << trsHead.XL << endl;
	}
	//***YL:Label of Y-axis   功耗的单位
	if (0 != trsHead.YL_length)
	{
		tag = 0x4A;
		outFile->write(&tag, 1);
		outFile->write((char*)&trsHead.YL_length, 1);
		outFile->write((char*)trsHead.YL, trsHead.YL_length);
		//cout << "trsHead.YL=" << trsHead.YL << endl;
	}
	//***XS:Scale value for X-axis 时间缩放率
	if (0 != trsHead.XS)
	{
		tag = 0x4B;
		length = 4;
		outFile->write(&tag, 1);
		outFile->write(&length, 1);
		outFile->write((char*)&trsHead.XS, length);
	}
	//***YS:Scale value for Y-axis 功耗缩放率
	if (0 != trsHead.YS)
	{
		tag = 0x4C;
		length = 4;
		outFile->write(&tag, 1);
		outFile->write(&length, 1);
		outFile->write((char*)&trsHead.YS, length);
	}
	//***TO:Trace offset for displaying trace numbers
	if (0 != trsHead.TO)
	{
		tag = 0x4D;
		length = 4;
		outFile->write(&tag, 1);
		outFile->write(&length, 1);
		outFile->write((char*)&trsHead.TO, length);
	}
	//***LS:Logarithmic scale
	if (0 != trsHead.LS)
	{
		tag = 0x4D;
		length = 1;
		outFile->write(&tag, 1);
		outFile->write(&length, 1);
		outFile->write((char*)&trsHead.LS, length);
	}

	//头部信息结束标志
	tag = 0x5F;
	outFile->write(&tag, 1);
	tag = 0x00;
	outFile->write(&tag, 1);
}

void Trace::readIndexTrace(TrsData* trsData, int index)
{
	//下标越界检查
	if (infile.eof() || index < 0)
	{
		cerr << "index out of trace two !";
		exit(0);
	}
	int sampleSize = 0;
	switch (SC_TYPE(this->trsHead.SC))
	{
	case FLOAT:
		sampleSize = sizeof(float);
		break;
	case INT:
		sampleSize = sizeof(int);
		break;
	case SHORT:
		sampleSize = sizeof(short);
		break;
	case BYTE:
		sampleSize = sizeof(char);
		break;
	default:
		break;
	}
	int offset = (index - currentTrace) * (this->trsHead.TS+ this->trsHead.DS + sampleSize * this->trsHead.NS);
	//文件指针偏移设置
	this->infile.seekg(offset, ios::cur);
	//下标越界检查
	if (infile.eof())
	{
		//改回原来的位置
		this->infile.seekg(-offset, ios::cur);
		cerr << "index out of trace two !";
		exit(0);
	}
	//设置当前曲线下标
	currentTrace = index;
	readNext(trsData);
}

void Trace::createTrace(const char* filename, TrsHead* trsHead, TrsData* trsData)
{
	ofstream outfile;
	outfile.open(filename, ios::out | ios::binary | ios::trunc);
	if (!outfile.is_open())
	{
		cerr << "Failed to open the file!";
		exit(0);
	}
	//文件指针调到首位

	/***************** trsfile header ************************/
	char length = 4;
	char tag = 0x41;
	//***NT:Number of traces  ->trace条数
	if (0 != trsHead->NT)
	{
		tag = 0x41;
		length = 4;
		outfile.write(&tag, 1);
		outfile.write(&length, 1);
		outfile.write((char*)&trsHead->NT, length);
	}
	//***NS:Number of samples per trace  ->每条trace的总点数
	if (0 != trsHead->NS)
	{
		tag = 0x42;
		length = 4;
		outfile.write(&tag, 1);
		outfile.write(&length, 1);
		outfile.write((char*)&trsHead->NS, length);
	}
	//***SC:Sample coding (e.g. type and length in bytes of each sample)  ->功耗的存储类型：float/int/short/byte
	if (0 != trsHead->SC)
	{
		tag = 0x43;
		length = 1;
		outfile.write(&tag, 1);
		outfile.write(&length, 1);
		outfile.write((char*)&trsHead->SC, length);
	}
	//***DS:Length of cryptographic data included in trace	->加/解密的明密文总数据个数，即：len(明文)+len(密文)
	if (0 != trsHead->DS)
	{
		tag = 0x44;
		length = 2;
		outfile.write(&tag, 1);
		outfile.write(&length, 1);
		outfile.write((char*)&trsHead->DS, length);
	}
	//***TS:Title space reserved per trace ->每条功耗的title
	if (0 != trsHead->TS)
	{
		tag = 0x45;
		length = 1;
		outfile.write(&tag, 1);
		outfile.write(&length, 1);
		outfile.write((char*)&trsHead->TS, length);
	}
	//***GT:Global trace title				->这组trace的title
	if (0 != trsHead->GT_length)
	{
		tag = 0x46;
		outfile.write(&tag, 1);
		outfile.write((char*)&trsHead->GT_length, 1);
		outfile.write((char*)trsHead->GT, trsHead->GT_length);
		//cout << "trsHead->GT=" << trsHead->GT << endl;
	}
	//***DC:Description   ->没有该tag
	if (0 != trsHead->DC_length)
	{
		tag = 0x47;
		outfile.write(&tag, 1);
		outfile.write((char*)&trsHead->DC_length, 1);
		outfile.write((char*)trsHead->DC, trsHead->DC_length);
		//cout << "trsHead->DC=" << trsHead->DC << endl;
	}
	//***XO:Offset in X-axis for trace representation	->没有该tag
	if (0 != trsHead->XO)
	{
		tag = 0x48;
		length = 4;
		outfile.write(&tag, 1);
		outfile.write(&length, 1);
		outfile.write((char*)&trsHead->XO, length);
	}
	//***XL:Label of X-axis   时间的单位
	if (0 != trsHead->XL_length)
	{
		tag = 0x49;
		outfile.write(&tag, 1);
		outfile.write((char*)&trsHead->XL_length, 1);
		outfile.write((char*)trsHead->XL, trsHead->XL_length);
		//cout << "trsHead->XL=" << trsHead->XL << endl;
	}
	//***YL:Label of Y-axis   功耗的单位
	if (0 != trsHead->YL_length)
	{
		tag = 0x4A;
		outfile.write(&tag, 1);
		outfile.write((char*)&trsHead->YL_length, 1);
		outfile.write((char*)trsHead->YL, trsHead->YL_length);
		//cout << "trsHead->YL=" << trsHead->YL << endl;
	}
	//***XS:Scale value for X-axis 时间缩放率
	if (0 != trsHead->XS)
	{
		tag = 0x4B;
		length = 4;
		outfile.write(&tag, 1);
		outfile.write(&length, 1);
		outfile.write((char*)&trsHead->XS, length);
	}
	//***YS:Scale value for Y-axis 功耗缩放率
	if (0 != trsHead->YS)
	{
		tag = 0x4C;
		length = 4;
		outfile.write(&tag, 1);
		outfile.write(&length, 1);
		outfile.write((char*)&trsHead->YS, length);
	}
	//***TO:Trace offset for displaying trace numbers
	if (0 != trsHead->TO)
	{
		tag = 0x4D;
		length = 4;
		outfile.write(&tag, 1);
		outfile.write(&length, 1);
		outfile.write((char*)&trsHead->TO, length);
	}
	//***LS:Logarithmic scale
	if (0 != trsHead->LS)
	{
		tag = 0x4D;
		length = 1;
		outfile.write(&tag, 1);
		outfile.write(&length, 1);
		outfile.write((char*)&trsHead->LS, length);
	}

	//头部信息结束标志
	tag = 0x5F;
	outfile.write(&tag, 1);
	tag = 0x00;
	outfile.write(&tag, 1);

	/***************** trsfile data ************************/
	for (int i = 0; i < trsHead->NT; i++)
	{
		if (0 != trsHead->TS)
		{
			outfile.write((char*)trsData[i].TSData, trsHead->TS);
		}
		if (0 != trsHead->DS)
		{
			outfile.write((char*)trsData[i].data, trsHead->DS);
		}
		switch (SC_TYPE(trsHead->SC))
		{
		case FLOAT:
			for (int j = 0; j < trsHead->NS; j++)
			{
				float value_float = 0;
				value_float = (float)(trsData[i].samples[j] / trsHead->YS);
				outfile.write((char*)&value_float, sizeof(float));
				//cout << "write：FLOAT=" << value_float << endl;
			}
			break;
		case INT:
			for (int j = 0; j < trsHead->NS; j++)
			{
				int value_int = 0;
				value_int = (int)(trsData[i].samples[j] / trsHead->YS);
				outfile.write((char*)&value_int, sizeof(int));
				//cout << "write：INT=" << value_int << endl;
			}
			break;
		case SHORT:
			for (int j = 0; j < trsHead->NS; j++)
			{
				short value_short = 0;
				value_short = (short)(trsData[i].samples[j] / trsHead->YS);
				outfile.write((char*)&value_short, sizeof(short));
				//cout << "write：SHORT=" << value_short << endl;
			}
			break;
		case BYTE:
			for (int j = 0; j < trsHead->NS; j++)
			{
				char value_byte = 0;
				value_byte = (char)(trsData[i].samples[j] / trsHead->YS);
				outfile.write((char*)&value_byte, sizeof(char));
				//cout << "write：BYTE=" << value_byte << endl;
			}
			break;
		default:
			break;
		}
		//cout << "write：BYTE=" << trsData[i].samples[0] / trsHead->YS << endl;
	}
	// outfile.close();
}



//void Trace::deleteTrace(const char* outFileName, const char* inFileName, int* trsIndexArray, int deleteNum)
//{
//	//读取文件
//	ifstream inFile;
//	inFile.open(inFileName, ios::in | ios::binary);
//	if (!inFile.is_open())
//	{
//		cerr << "Failed to open the file!";
//		exit(0);
//	}
//	TrsHead trsHead;
//	readHeard(&inFile, &trsHead);
//	//写入文件
//	ofstream outFile;
//	outFile.open("delete_temp.trs", ios::out | ios::binary | ios::trunc);
//	if (!outFile.is_open())
//	{
//		cerr << "Failed to open the file!";
//		exit(0);
//	}
//	/***************** trsfile data ************************/
//	TrsData trsData;
//	int currentTrace = 0;
//	int count_del = 0;
//	for (int i = 0; i < trsHead.NT; i++)
//	{
//		int flag_del = 0;
//		readIndexTrace(&inFile, &trsData, trsHead, i,0);
//		for (int j = count_del; j < deleteNum; j++)
//		{
//			if (i == trsIndexArray[j])
//			{
//				flag_del = 1;
//				//交换删除数组的值，已经删除的放前面，可以减少循环判断
//				int temp = trsIndexArray[j];
//				trsIndexArray[j] = trsIndexArray[0];
//				trsIndexArray[0] = temp;
//				//已删除的计数+1
//				count_del++;
//			}
//		}
//		if (0 == flag_del)
//		{
//			writeNext(&outFile, &trsData, trsHead);
//		}
//	}
//	writeHead(&outFile, trsHead);
//	inFile.close();
//	outFile.close();
//	rename("delete_temp.trs", outFileName);
//	remove("delete_temp.trs");//删除文件
//}


void Trace::saveTraceArea( const char* outFileName, const char* inFileName, int startPoint, int length)
{
	//"D:\\MathMagic\\Detector\\TempTracesCMSIS_conv1_25.trs"
	Trace trace(inFileName);
    TrsData trsData ;
    TrsHead outTrsHead ;
	TrsData outTrsData ;
	outTrsData.samples=(float*)malloc(length*sizeof(float));
	printf("done1");
	outTrsData.data = (uint8_t*)malloc(outTrsHead.DS*sizeof(uint8_t));
	printf("done2");
	outTrsData.TSData =(uint8_t*)malloc(outTrsHead.TS*sizeof(uint8_t));		
	printf("done3");
	std::memset(outTrsData.samples, 0, length * sizeof(float));
	std::memset(outTrsData.data, 0, outTrsHead.DS * sizeof(uint8_t));		
	std::memset(outTrsData.TSData, 0, outTrsHead.TS * sizeof(uint8_t));
    
	printf("done4");
	outTrsHead = trace.trsHead;
	outTrsHead.NT = trace.trsHead.NT;
	outTrsHead.NS = length;

	ofstream outfile;
	outfile.open(outFileName, ios::out | ios::binary | ios::trunc);
    Trace::writeHead(&outfile, outTrsHead);
	printf("done!\n");

	for (int i = 0; i < outTrsHead.NT; i++){
	// for (int i = 0; i < 10; i++){
		trace.readNext(&trsData);
		//拷贝明密文到结果数据
		std::memcpy(outTrsData.data, trsData.data, outTrsHead.DS);
		std::memcpy(outTrsData.TSData, trsData.TSData, outTrsHead.TS);
		// std::memcpy(outTrsData[i].samples, trsData->samples, outTrsHead->NS * sizeof(float));
		for(int j=0;j<length;j++){
			outTrsData.samples[j]=trsData.samples[startPoint+j];
		}
		Trace::writeNext(&outfile, &outTrsData, outTrsHead);
		// printf("%d,",i);
	}
	
	free(outTrsData.samples);
	free(outTrsData.data);
	free(outTrsData.TSData);
	printf("50000");
	outfile.close();
	// return ;
}


Trace::~Trace() {
	infile.close();
	delete this->trsHead.GT;
	this->trsHead.GT = NULL;
	delete this->trsHead.DC;
	this->trsHead.DC = NULL;
	delete this->trsHead.XL;
	this->trsHead.XL = NULL;
	delete this->trsHead.YL;
	this->trsHead.YL = NULL;
}
