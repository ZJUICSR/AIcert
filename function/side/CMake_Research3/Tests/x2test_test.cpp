#include "../Inc/interface.h"




int main(int argc, char const *argv[])
{
    #if 0
    char inFile[]="Trs/samples/cpa/elmotrace47/(21.108.112.47)elmotracegaus_cpa_47.trs";
    char outFile[]="Trs/samples/cpa/elmotrace47/x2test_out47.txt";
    #endif

    #if 1
    char inFile[]="Trs/samples/cpa/elmotrace-9/(0.108.112.-9)elmotracegaus_cpa_-9.trs";
    char outFile[]="Trs/samples/cpa/elmotrace-9/x2test_out-9.trs";
    #endif
    
    x2test_(inFile, outFile);
    return 0;

}