.. meta::
   :robots: index,follow
   :description: libumem documentation

.. sectionauthor:: Pearu Peterson <pearu.peterson at quansight.com>

.. default-domain:: cpp
                   
libumem public C++ API
======================

Memory location
---------------

Within libumem C++ API, the data location address is a
:class:`umem::Address` object that captures in it also the memory
context information.

The generated libumem C++ API is available here__ .

__ https://codedocs.xyz/plures/umem/

Examples
--------

The following program illustrates the usage of libumem as a
replacement of :file:`stdlib.h` malloc/free functionality.

.. code-block:: cpp

   #include "umem.h"

   int main()
   {
     umem::Host host;
     
     {
       // allocate a length 10 array of doubles
       umem::Address adr = host.calloc(sizeof(double), 10);  

       // application specific code follows, for instace, initialize the array
       // as range(10):
       double * ptr = (double*)adr;
       for(int i=0; i<10; ++i) ptr[i] = (double)i;

       // leaving the scope frees the adr memory
     }
     
     // leaving the scope destructs host
   }

The following program illustrates the synchronization of data between
host RAM and GPU device memory:

.. code-block:: cpp

   #include "umem.h"

   int main()
   {
     umem::Host host;            // construct host RAM context
     umem::Cuda cuda(0);         // construct GPU device 0 context

     {
       // allocate a length 10 array of doubles in GPU device aligned in
       // 128 byte boundaries
       size_t cuda_alignment = 128;
       umem::Address cuda_adr = cuda.aligned_alloc(cuda_alignment, 10*sizeof(double));  

       // establish a connection between host and GPU memories.
       // for allocated host buffer, we'll use alignment 64
       size_t host_alignment = 64;
       umem::Address host_adr = cuda_adr.connect(10*sizeof(double), host_alignment);
     
       // application specific code, for instace, initialize the array
       // as range(10):
       double * ptr = (double*)host_adr;
       for(int i=0; i<10; ++i) ptr[i] = (double)i;
       host_adr.sync(10*sizeof(double));
       // now the GPU device memory is initialized as range(10)

       // say, the GPU device changed the allocated data, so we sync the
       // data to host buffer:
       host_adr.update(10*sizeof(double));

       // leaving the scope frees host_adr and cuda_adr
     }

     // leaving the scope destructs cuda and host
   }

Note that the only device specific lines in the above example are the
constructor calls. The code that follows the constructor calls, are
device independent and would function exactly the same when, say,
swapping the :expr:`host` and :expr:`cuda` variables.

