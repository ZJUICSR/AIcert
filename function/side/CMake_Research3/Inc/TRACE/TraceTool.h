#include "Trace.h"
class TraceTools
{
public:
	static Trs addTrace(Trs trs1, Trs trs2);
	static Trs subTrace(Trs trs1, Trs trs2);
	static TrsData meanTrace(Trs trs);
	static void meanTrace(TrsData* outTrsData, TrsHead trsHead, TrsData* TrsData);
	static void trs2txt(const char* trsFileName);
	static void trs2txt(char* trsFileName, int txtMessageLength, int txtCipherLength);
	//void trsD_input(TrsData trsdata, int bias);
	static void pca(char* fileName, char *outfileName);
	//spa
	static void cor(TrsData *baseTrsData, TrsData *srcTrsData, int samplePointNum, int length, int immp, TrsHead trsHead, const char *trsfileout);
};

