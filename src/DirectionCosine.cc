#include "DirectionCosine.hh"
#include "MC_RNG_State.hh"
#include "PhysicalConstants.hh"

#ifdef HAVE_SYCL
#include <CL/sycl.hpp>
#define SIN  sycl::sin
#define COS  sycl::cos
#define SQRT sycl::sqrt
#define LOG  sycl::log
#else
#include <cmath>
#define SIN  sin
#define COS  cos
#define SQRT sqrt
#define LOG  log
#endif

void DirectionCosine::Sample_Isotropic(uint64_t *seed)
{
    this->gamma  = 1.0 - 2.0*rngSample(seed);
    double sine_gamma  = SQRT((1.0 - (gamma*gamma)));
    double phi         = PhysicalConstants::_pi*(2.0*rngSample(seed) - 1.0);

    this->alpha  = sine_gamma * COS(phi);
    this->beta   = sine_gamma * SIN(phi);
}
