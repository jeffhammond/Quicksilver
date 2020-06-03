#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "cudaFunctions.hh"
#include "cudaUtils.hh"

namespace
{
#if HAVE_CUDA
#include "cudaFunctions.hh"
    void WarmUpKernel(sycl::nd_item<3> item_ct1)
    {
   int global_index = getGlobalThreadID(item_ct1);
        if( global_index == 0)
        {
        }
        item_ct1.barrier();
    }
#endif
}

#if defined (HAVE_CUDA)
void warmup_kernel()
{
   dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) { WarmUpKernel(item_ct1); });
   });
   dpct::get_current_device().queues_wait_and_throw();
}
#endif

#if defined (HAVE_CUDA)
int ThreadBlockLayout(sycl::range<3> &grid, sycl::range<3> &block,
                      int num_particles)
{
    int run_kernel = 1;
    const uint64_t max_block_size = 65535;
    const uint64_t threads_per_block = 128;

   block[0] = threads_per_block;
   block[1] = 1;
   block[2] = 1;

    uint64_t num_blocks = num_particles / threads_per_block + ( ( num_particles%threads_per_block == 0 ) ? 0 : 1 );

    if( num_blocks == 0 )
    {
        run_kernel = 0;
    }
    else if( num_blocks <= max_block_size )
    {
      grid[0] = num_blocks;
      grid[1] = 1;
      grid[2] = 1;
    } 
    else if( num_blocks <= max_block_size*max_block_size )
    {
      grid[0] = max_block_size;
      grid[1] = 1 + (num_blocks / max_block_size);
      grid[2] = 1;
    }
    else if( num_blocks <= max_block_size*max_block_size*max_block_size )
    {
      grid[0] = max_block_size;
      grid[1] = max_block_size;
      grid[2] = 1 + (num_blocks / (max_block_size * max_block_size));
    }
    else
    {
        printf("Error: num_blocks exceeds maximum block specifications. Cannot handle this case yet\n");
        run_kernel = 0;
    }

    return run_kernel;
} 
#endif

#if defined (HAVE_CUDA)
SYCL_EXTERNAL DEVICE int getGlobalThreadID(sycl::nd_item<3> item_ct1)
{
   int blockID = item_ct1.get_group(2) +
                 item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                 item_ct1.get_group(0) * item_ct1.get_group_range(2) *
                     item_ct1.get_group_range(1);

   int threadID =
       blockID * (item_ct1.get_local_range().get(2) *
                  item_ct1.get_local_range().get(1) *
                  item_ct1.get_local_range().get(0)) +
       item_ct1.get_local_id(0) * (item_ct1.get_local_range().get(2) *
                                   item_ct1.get_local_range().get(1)) +
       item_ct1.get_local_id(1) * item_ct1.get_local_range().get(2) +
       item_ct1.get_local_id(2);
    return threadID;
}
#endif
