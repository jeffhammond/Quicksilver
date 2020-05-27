#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#ifdef DPCPP_COMPATIBILITY_TEMP
#define qs_assert( cond) \
   do \
   { \
      if (!(cond)) \
      { \
        printf("ERROR\n"); \
      } \
   } while(0)
#else
#define qs_assert( cond)                        \
   do \
   { \
      if (!(cond)) \
      { \
        printf("file=%s: line=%d ERROR\n",__FILE__,__LINE__); \
      } \
   } while(0)
#endif
