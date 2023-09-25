#include "../Inc/interface.h"




int main(int argc, char const *argv[])
{
    
    #if 1 //47
    char inFile[]="Trs/samples/hpa/elmotrace47/(21.108.112.47)elmotracegaus_hpa_47.trs";
    char outFile[]="Trs/samples/hpa/elmotrace47/hpa_out47.txt";
    #endif

    #if 0 //-9
    char inFile[]="Trs/samples/hpa/elmotrace-9/(0.112.116.-9)elmotracegaus_hpa_-9.trs";
    char outFile[]="Trs/samples/hpa/elmotrace-9/hpa_out-9.txt";
    #endif

    #if 0 //2
    char inFile[]="Trs/samples/hpa/elmotrace2/(75.112.116.2)elmotracegaus_hpa_2.trs";
    char outFile[]="Trs/samples/hpa/elmotrace2/hpa_out2.txt";
    #endif

    hpa_(inFile, outFile);
    

    return 0;

}