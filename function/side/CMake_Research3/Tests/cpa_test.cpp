#include "../Inc/interface.h"




int main(int argc, char const *argv[])
{
    #if 1
    char* inFile = "Trs/samples/cpa/elmotrace-9/(0-108-112)elmotracegaus_cpa_-9.trs";
    char* outFile = "Trs/samples/cpa/elmotrace-9/cpa_out-9.txt";
    cpa_(inFile, outFile);
    #endif

    return 0;

}