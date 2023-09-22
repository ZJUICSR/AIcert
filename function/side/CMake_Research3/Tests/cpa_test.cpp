#include "../Inc/interface.h"




int main(int argc, char const *argv[])
{
    #if 1 //47
    char inFile[]="Trs/samples/cpa/elmotrace47/(2122.108.112.47)elmotracegaus_cpa_47.trs";
    char outFile[]="Trs/samples/cpa/elmotrace47/cpa_out47.txt";
    #endif

    #if 0 //-9
    char inFile[]="Trs/samples/cpa/elmotrace-9/(022.112.116.-9)elmotracegaus_cpa_-9.trs";
    char outFile[]="Trs/samples/cpa/elmotrace-9/cpa_out-9.txt";
    #endif

    #if 0 //2
    char inFile[]="Trs/samples/cpa/elmotrace2/(7522.112.116.2)elmotracegaus_cpa_2.trs";
    char outFile[]="Trs/samples/cpa/elmotrace2/cpa_out2.txt";
    #endif

    cpa_(inFile, outFile);

    return 0;

}