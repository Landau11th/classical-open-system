#include "rand64.hpp"


using namespace Deng::RandNumGen;

unsigned long long int LCG64::x;
double LCG64::drand64()
{
    x = 6364136223846793005ll * x + (long long int) 1;
    return (double) x * 5.4210108624275218e-20;
}
void LCG64::srand64(int seed)//, FILE *fp)
{
    assert(sizeof(long long int) == 8);
    x = seed;

    //fprintf(fp, "drand 64-bit, initial seed x = %d\n", seed);
}
