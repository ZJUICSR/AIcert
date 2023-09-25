#include "../Inc/interface.h"

int main(int argc, char const *argv[])
{
    char inFile[]="Trs/samples/spa/50-all-1.trs";
    char outFile[]="Trs/samples/spa/results/";

    spa_(inFile, outFile);
    return 0;
}
