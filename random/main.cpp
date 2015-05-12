#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <complex>
#include <cstdlib>
#include <vector>
#include "random.h"

using namespace std;

int main(){

RanGSL rand(1234);

for( int i=0; i<100; i++)
   cout << rand() << endl;

return 0;
}

