#ifndef MEMORY_CONTROL_HH
#define MEMORY_CONTROL_HH

#include <cstdio>
#include <cstdlib>

#include <CL/sycl.hpp>

#include "cudaUtils.hh"

#include "qs_assert.hh"

#include "QS_sycl.hh"

namespace MemoryControl
{
   enum AllocationPolicy {HOST_MEM, UVM_MEM, UNDEFINED_POLICY};

   template <typename T>
   T* allocate(const int size, const AllocationPolicy policy)
   {
      if (size == 0) { return NULL;}
      T* tmp = NULL;

      switch (policy)
      {
        case AllocationPolicy::HOST_MEM:
         tmp = new T [size];
         break;
        case AllocationPolicy::UVM_MEM:
         void * ptr;
         ptr = (void *)sycl::malloc_shared(size * sizeof(T), q);
         tmp = new(ptr) T[size];
         break;
        default:
         qs_assert(false);
         break;
      }
      return tmp;
   }

   template <typename T>
   void deallocate(T* data, const int size, const AllocationPolicy policy)
   {
      switch (policy)
      {
        case MemoryControl::AllocationPolicy::HOST_MEM:
         delete[] data;
         break;
        case UVM_MEM:
         for (int i=0; i < size; ++i) data[i].~T();
         sycl::free(data, q);
         break;
        default:
         qs_assert(false);
         break;
      }
   }
}


#endif
