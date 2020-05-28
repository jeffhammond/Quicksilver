#ifndef INIT_MC_HH
#define INIT_MC_HH

#include <cstdio>

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <tuple>

using std::vector;
using std::string;
using std::set;
using std::cout;
using std::endl;
using std::map;
using std::make_pair;

class Parameters;
class MonteCarlo;

MonteCarlo* initMC(const Parameters& params);

#endif
