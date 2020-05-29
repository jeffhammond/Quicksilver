#ifndef DECLAREMACRO_HH
#define DECLAREMACRO_HH

#ifdef HAVE_SYCL
    #define HOST_DEVICE SYCL_EXTERNAL
    #define HOST_DEVICE_CUDA SYCL_EXTERNAL
    #define HOST_DEVICE_CLASS
    #define HOST_DEVICE_END
    #define DEVICE SYCL_EXTERNAL
    #define DEVICE_END
    #define HOST_END
    #define GLOBAL
#elif defined(HAVE_CUDA)
    #define HOST_DEVICE
    #define HOST_DEVICE_CUDA
    #define HOST_DEVICE_CLASS
    #define HOST_DEVICE_END
    #define DEVICE
    #define DEVICE_END
    //#define HOST __host__
    #define HOST_END
    #define GLOBAL
#elif defined(HAVE_OPENMP_TARGET)
    #define HOST_DEVICE _Pragma( "omp declare target" )
    #define HOST_DEVICE_CUDA
    #define HOST_DEVICE_CLASS _Pragma( "omp declare target" )
    #define HOST_DEVICE_END _Pragma("omp end declare target")
    //#define HOST_DEVICE #pragma omp declare target
    //#define HOST_DEVICE_END #pragma omp end declare target
    //#define DEVICE #pragma omp declare target
    //#define DEVICE_END #pragma omp end declare target
    //#define HOST
    #define HOST_END
    #define GLOBAL
#else
    #define HOST_DEVICE
    #define HOST_DEVICE_CUDA
    #define HOST_DEVICE_CLASS
    #define HOST_DEVICE_END
    #define DEVICE
    #define DEVICE_END
    //#define HOST
    #define HOST_END
    #define GLOBAL
#endif

#endif
