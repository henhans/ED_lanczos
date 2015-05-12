#ifndef MY_UTILS
#define MY_UTILS
#include <cmath>
#include <string>
#include <sstream>
#include <iostream>

template <class T>
T power(T p, int k)
{
  T s = 1;
  for (int i=0; i<k; i++) s*=p;
  return s;
}


#endif
