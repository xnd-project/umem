.. meta::
   :robots: index,follow
   :description: libumem documentation

.. sectionauthor:: Pearu Peterson <pearu.peterson at quansight.com>

.. default-domain:: c

libumem public C API
====================

Memory location
---------------

Within libumem C API, the data location address is a :type:`uintptr_t`
value. In the case of host RAM, the address value is equal to data
pointer value. For other storage devices, the address value may have
various interpretations that depends on the storage device as well as
the storage device driver library. However, the fundamental assumption
of address value is that its increments give valid addresses of the
whole data content stored in the device.

Examples
--------

The following program illustrates the usage of libumem as a
replacement of :file:`stdlib.h` malloc/free functionality.

.. code-block::

   #include "umem.h"

   int main()
   {
     umemHost host;
     umemHost_ctor(&host);    // construct host RAM context

     // allocate a length 10 array of doubles
     uintptr_t adr = host.calloc(sizeof(double), 10);  

     // application specific code follows, for instace, initialize the array
     // as range(10):
     double * ptr = (double*)adr;
     for(int i=0; i<10; ++i) ptr[i] = (double)i;

     // free the allocated memory area
     umem_free(&host, adr);
     umem_dtor(&host);        // destruct host RAM context
   }

The following program illustrates the synchronization of data between
host RAM and GPU device memory:

.. code-block::

   #include "umem.h"

   int main()
   {
     umemHost host;
     umemCuda cuda;
     umemHost_ctor(&host);       // construct host RAM context
     umemCuda_ctor(&cuda, 0);    // construct GPU device 0 context

     // allocate a length 10 array of doubles in GPU device aligned in
     // 128 byte boundaries
     size_t cuda_alignment = 128;
     uintptr_t cuda_adr = cuda.aligned_alloc(cuda_alignment, 10*sizeof(double));  

     // establish a connection between host and GPU memories.
     // for allocated host buffer, we'll use alignment 64
     size_t host_alignment = 64;
     uintptr_t host_adr = umem_connect(&cuda, cuda_adr,
                                       10*sizeof(double),
                                       &host, host_alignment);
     
     // application specific code, for instace, initialize the array
     // as range(10):
     double * ptr = (double*)host_adr;
     for(int i=0; i<10; ++i) ptr[i] = (double)i;
     umem_sync_from(&cuda, cuda_adr, &host, host_adr, 10);
     // now the GPU device memory is initialized as range(10)

     // say, the GPU device changed the allocated data, so we sync the
     // data to host buffer:
     umem_sync_to(&cuda, cuda_adr, &host, host_adr, 10);
     
     // disconnect the host and GPU device memories, this also frees host buffer
     umem_disconnect(&cuda, cuda_adr, &host, host_adr, host_alignment);
     
     // free the allocated memory area in the GPU device
     umem_aligned_free(&cuda, cuda_adr);
     
     umem_dtor(&cuda);        // destruct GPU device context
     umem_dtor(&host);        // destruct host RAM context
   }

Note that the only device specific lines in the above example are the
constructor calls. The code that follows the constructor calls, are
device independent and would function exactly the same when, say,
swapping the :data:`host` and :data:`cuda` variables.


Supported storage devices
-------------------------

The libumem C-API provides the following device memory context
objects (C :type:`struct` instances):

* :type:`umemHost` - `stdlib.h` based interface to host RAM,

* :type:`umemFile` - `stdio.h` based interface to files,

* :type:`umemCuda` - CUDA based interface to GPU device memory.

Each device memory context has specific initializer (a
constructor). However, all other memory management methods such as
destructors and copying tools are universal among the all memory
storage devices.

:type:`umemHost` context
''''''''''''''''''''''''

The :type:`umemHost` type defines a host RAM context and it must be
initialized using the constructor function :func:`umemHost_ctor`:

.. code-block:: c

   void umemHost_ctor(umemHost * const this);

To destruct the host RAM context object, use :func:`umem_dtor`
destructor function. See below.

:type:`memFile` context
'''''''''''''''''''''''

The `umemFile` type defines a file context that must be initialized
with the following constructor function:

.. code-block:: c

   void umemFile_ctor(umemFile * const ctx, const char * filename, const char * mode);

Here :data:`filename` is the path name of a file that is opened using
given :data:`mode`. The :data:`mode` string must start with one of the
following strings: ``"r"``, ``"r+"``, ``"w"``, ``"w+"``, ``"a"``,
``"a+"``. The :data:`mode` string may include also the character
``'b'`` to indicate binary file content.

The destructor function :func:`umem_dtor` closes the file.


:type:`memCuda` context
'''''''''''''''''''''''

The :type:`umemCuda` type defines a CUDA based GPU device memory
context that must be initialized with the following constructor
function:

.. code-block:: c

   void umemCuda_ctor(umemCuda * const ctx, int device);

Here :data:`device` is GPU device number. The constructor function will set
the corresponding GPU device.

While the destructor function :func:`umem_dtor` does not call any CUDA
API functions, it is recommended to use it to destruct :type:`umemCuda`
objects after it is not needed anymore.

Universal API methods
---------------------

Desctructor
'''''''''''

.. code-block:: c

   void umem_dtor(void const * ctx);

Destructs given memory context.

Memory allocation/deallocation
''''''''''''''''''''''''''''''

.. code-block:: c

   uintptr_t umem_alloc(void const * ctx, size_t nbytes);

Allocates :data:`nbytes` of memory in the given storage device. The
allocated memory is uninitialized.

.. code-block:: c

   uintptr_t umem_calloc(void const * ctx, size_t nmemb, size_t size);

Allocated an array of given :data:`size` and member byte size
:data:`nmemb`. Returns the starting address of allocated memory. The
allocated memory is zero-initialized.

.. code-block:: c

   void umem_free(void const * ctx, uintptr_t adr);

Frees the memory that was allocated with methods
:func:`umem_alloc` or :func:`umem_calloc`.

.. code-block:: c

   uintptr_t umem_aligned_alloc(void const * ctx, size_t alignement, size_t size);

Allocates :data:`size` bytes (plus some extra) of device memory so
that the returned starting address is aligned to given
:data:`alignement` value.

.. code-block:: c

   uintptr_t umem_free_aligned(void const * ctx, uintptr_t adr);

Frees the memory that was allocated with methods
:func:`umem_aligned_alloc`.

Memory initialization
'''''''''''''''''''''

For initializing device memory with arbitrary data from host RAM, see
below how to copy data between devices.

.. code-block:: c

   uintptr_t umem_set(void const * ctx, uintptr_t adr, int c, size_t nbytes);

Sets :data:`nbytes` of device memory with starting address :data:`adr`
to byte value :data:`c` (the memory area will be filled byte-wise).

Copying data between memory devices
'''''''''''''''''''''''''''''''''''

.. code-block:: c

   void umem_copy_to(void * const src_ctx, uintptr_t src_adr,
                     void * const dest_ctx, uintptr_t dest_adr,
                     size_t nbytes);

Copies :data:`nbytes` of source device memory starting at address
:data:`src_adr` to destiniation device memory starting at address
:data:`dest_adr`.  The source and destination memory devices can be
different or the same. When the source and destination devices are the
same then the copying areas should not overlap, otherwise the result
will be undetermined.

.. code-block:: c

   void umem_copy_from(void * const dest_ctx, uintptr_t dest_adr,
                       void * const src_ctx, uintptr_t src_adr,
                       size_t nbytes);

The inverse of :func:`umem_copy_to`.

.. code-block:: c

   void umem_copy_to_safe(void * const src_ctx, uintptr_t src_adr, size_t src_size,
                          void * const dest_ctx, uintptr_t dest_adr, size_t dest_size,
                          size_t nbytes);
   void umem_copy_from_safe(void * const dest_ctx, uintptr_t dest_adr, size_t dest_size,
                            void * const src_ctx, uintptr_t src_adr, size_t src_size,
                            size_t nbytes);

These methods have the same functionality as :func:`umem_copy_to` and
:func:`umem_copy_from` but include checking the bounds of copying
areas. The :data:`src_size` and :data:`dest_size` are the memory area
widths within the copying process is expected to be carried
out. Usually the widths correspond to the size of allocated areas but
not necessarily, for instance, when copying subsets of the allocated
area.

When the copying process would go out of bounds, e.g. when
``max(src_size, dest_size) < nbytes``, then :cpp:enum:`umemIndexError`
is set as the status value in the problematic device context and the
functions will return without starting the copying process.

Keeping data in sync between memory devices
'''''''''''''''''''''''''''''''''''''''''''

.. code-block:: c

   uintptr_t umem_connect(void * const src_ctx, uintptr_t src_adr,
                          size_t nbytes,
                          void * const dest_ctx, size_t dest_alignment);

Establishes a connection between the two memory devices and returns
the paired address in the destination context.

When the memory devices are different or when the source alignement
does not match with :data:`dest_alignment` then :data:`nbytes` of
memory is allocated in destination context and the paired address will
be the starting address of allocated memory. Otherwise :data:`src_adr`
will be returned as the paired address.

.. code-block:: c

   void umem_disconnect(void * const src_ctx, uintptr_t src_adr,
                        void * const dest_ctx, uintptr_t dest_adr,
                        size_t dest_alignment)

Disconnect the two devices that were connected using
:func:`umem_connect` function, that is, free the memory that
:func:`umem_connect` may have been allocated. The :data:`dest_adr`
must be the paired address returned previously by :func:`umem_connect`
and the other arguments must be the same that was used to call
:func:`umem_connect`.

.. code-block:: c

   void umem_sync_to(void * const src, uintptr_t src_adr,
                     void * const dest, uintptr_t dest_adr, size_t nbytes);
   void umem_sync_from(void * const dest, uintptr_t dest_adr,
                     void * const src, uintptr_t src_adr, size_t nbytes);

Syncronize the data between the two devices. When the source and
destination devices are the same and ``src_adr == dest_adr`` then
:func:`umem_sync_to` and :func:`umem_sync_from` are NOOP.

Note that :data:`nbytes` must be less or equal to :data:`nbytes` value
that were using in calling :func:`umem_connect` function.

.. code-block:: c

   void umem_sync_to_safe(void * const src_ctx, uintptr_t src_adr, size_t src_size,
                          void * const dest_ctx, uintptr_t dest_adr, size_t dest_size,
                          size_t nbytes);
   void umem_sync_from_safe(void * const dest_ctx, uintptr_t dest_adr, size_t dest_size,
                            void * const src_ctx, uintptr_t src_adr, size_t src_size,
                            size_t nbytes);

These functions have the same functionality as :func:`umem_sync_to`
and :func:`umem_sync_from` but include checking the bounds of
synchronizing memory areas. Same rules apply as in
:func:`umem_copy_to_save` and :func:`umem_copy_from_save`, see above.
                            
Status message handling
-----------------------

The success or failure of calling libumem C-API methods described
above can be determined by checking the status of memory context
objects that participated in the call.

.. code-block:: c

   bool umem_is_ok(void * const ctx);

Returns :data:`true` if the memory context experienced no failures.

.. code-block:: c

   umemStatusType umem_get_status(void * const ctx);

Returns the status flag from the memory context object.

.. code-block:: c

   const char * umemStatusType umem_get_message(void * const ctx);

Returns the status message from the memory context object. It will be
empty string ``""`` when no message has been set (e.g. when
:func:`umem_is_ok` returns :data:`true`).

.. code-block:: c

   void umem_set_status(void * const ctx,
                        umemStatusType type, const char * message);

Sets the status :data:`type` and status :data:`message` to given
memory context object. Use this function when you want to propagate
the exceptions raised by libumem C-API methods with extra messages to
a caller function that will handle the exceptions.

Note that :func:`umem_set_status` overwrites the previouly set status
type, however, the status message will appended to the previouly set
status message. The overwrite of status type will be recorded in
status message as well.

.. code-block:: c

   void umem_clear_status(void * const ctx)

Clears memory context object status content: sets the status to "OK"
and clears status messages. One should call :func:`umem_clear_status`
after handling any exceptions raised by the libumem C-API methods.


Utility functions
-----------------

The following utility functions are used internally in libumem but
might be useful for application programs as well.

.. code-block:: c

   const char* umem_get_status_name(umemStatusType type);

Returns status :func:`type` as a string.

.. code-block:: c

   inline const char* umem_get_device_name(void * const ctx);

Returns the name of memory context as a string.

.. code-block:: c

   bool umem_is_same_device(void * const one_ctx, void * const other_ctx);

Returns :data:`true` when the memory context objects represent the
same memory storage device, that is, the addresses of both devices
will be comparable.

.. code-block:: c

   uintptr_t umem_aligned_origin(void const * ctx, uintptr_t adr);

Return the original memory address that was obtained when allocating
device memory with :func:`umem_aligned_alloc`.


libumem internal C API
======================

This section is for developers who want to extend libumem with other
memory storage devices or want to understand libumem sources.

libumem design
--------------

While libumem is implemented in C, it uses OOP design. This design
choice simplifies exposing libumem to other programming languages that
support OOP, such as C++, Python, etc, not to mention the advantages
of using OOP to implement abstract view of variety of data storage
devices in an unified way.

:type:`umemVirtual` base type
-----------------------------

A data storage device is representes as memory context type that is
derived from :type:`umemVirtual` type:

.. code-block:: c

   typedef struct {
     struct umemVtbl const *vptr;
     umemDeviceType type;
     umemStatus status;
     void* host;
   } umemVirtual;

The member :data:`vprt` is a pointer to virtual table of methods. This
table will be filled in with device specific methods in the
constructors of the correspondig derived types:

.. code-block:: c

   struct umemVtbl {
     void (*dtor)(umemVirtual * const ctx);
     bool (*is_same_device)(umemVirtual * const ctx, umemVirtual * const other_ctx);
     uintptr_t (*alloc)(umemVirtual * const ctx, size_t nbytes);
     uintptr_t (*calloc)(umemVirtual * const ctx, size_t nmemb, size_t size);
     void (*free)(umemVirtual * const ctx, uintptr_t adr);
     uintptr_t (*aligned_alloc)(umemVirtual * const this, size_t alignment, size_t size);
     uintptr_t (*aligned_origin)(umemVirtual * const this, uintptr_t aligned_adr);
     void (*aligned_free)(umemVirtual * const this, uintptr_t aligned_adr);
     void (*set)(umemVirtual * const this, uintptr_t adr, int c, size_t nbytes);
     void (*copy_to)(umemVirtual * const this, uintptr_t src_adr,
                     umemVirtual * const that, uintptr_t dest_adr,
		     size_t nbytes);
     void (*copy_from)(umemVirtual * const this, uintptr_t dest_adr,
                       umemVirtual * const that, uintptr_t src_adr,
                       size_t nbytes);
   };

The descriptions of members methods are as follows: 

:func:`dtor`
      A destructor of memory context. It should clean-up any resources
      that are allocted in the memory constructor.

:func:`is_same_device`
      A predicate function that should return :data:`true` when the
      memory context objects referenced by :type:`ctx` and
      :type:`other_ctx` are the same, that is, the memory context
      objects would allocate memory in the same data address space.

:func:`alloc`, :func:`calloc`, :func:`free`
      Device memory allocator and deallocation functions. The
      allocator functions must return starting address of the
      allocated memory area. The :func:`free` function must deallocate
      the corresponding memory.

:func:`aligned_alloc`, :func:`aligned_free`, :func:`aligned_origin`
      Device memory allocator and deallocation functions with
      specified alignment. The :func:`aligned_origin` will return the
      orignal address of allocated memory. As a rule, the address
      returned by :func:`aligned_alloc` points to a memory area that
      is a subset of memory area starting at the address returned by
      :func:`aligned_origin`.

:func:`set`
      A function that must initialize memory content with given byte
      value in :data:`c`.

:func:`copy_to`, :func:`copy_from`
      Functions for copying data from one memory context to another
      memory context.  If the storage device driver does not support
      copying data to another storage device, one can use host RAM as
      a buffer. It is assumed that the storage device always supports
      copying data between the device memory and host RAM memory.
      
The member :data:`type` specifies the memory device type defined in
:cpp:enum:`umemDeviceType` enum.

The member :data:`status` holds the status information of given memory
context as :type:`umemStatus` type:

.. code-block:: c

   typedef struct {
     umemStatusType type;
     char* message;
   } umemStatus;

Finally, the member :data:`host` holds a pointer to :type:`umemHost`
object that is used to allocate/deallocate intermediate memory buffers
that the storage device specific methods might need.

   
Adding a new data storage device support to libumem
---------------------------------------------------

In the following, the required steps of addning new data storage
device support are described. To be specific, let's assume that we
want to add support to a data storage device called "MyMem".

Defining new type :type:`umemMyMem`
'''''''''''''''''''''''''''''''''''

The template for defining a new memory context type is

.. code-block:: c
   
   typedef struct {
     umemVirtual super;  /// REQUIRED
     umemHost host;      /// REQUIRED
     // Define device specific members:
     ...                 /// OPTIONAL
   } umemMyMem;

The :type:`umemMyMem` must be defined in :file:`umem.h`.

Adding new device type to :cpp:enum:`umemDeviceType`
''''''''''''''''''''''''''''''''''''''''''''''''''''

Add new item :data:`umemMyMemDevice` to :cpp:enum:`umemDeviceType`
enum definition in :file:`umem.h`.

Defining constructor function :func:`umemMyMem_ctor`
''''''''''''''''''''''''''''''''''''''''''''''''''''

The constructor function of memory context must initialize the virtual
table of methods and other members in :type:`umemMyMem`. The template
for the constructor function is

.. code-block:: c

   void umemMyMem_ctor(umemMyMem * const ctx,
                       /* device specific parameters: */ ... )
   {
     static struct umemVtbl const vtbl = {
       &umemMyMem_dtor_,
       &umemMyMem_is_same_device_,
       &umemMyMem_alloc_,
       &umemVirtual_calloc,
       &umemMyMem_free_,
       &umemVirtual_aligned_alloc,
       &umemVirtual_aligned_origin,
       &umemVirtual_aligned_free,
       &umemMyMem_set_,
       &umemMyMem_copy_to_,
       &umemMyMem_copy_from_,
     };
     umemHost_ctor(&ctx->host);                   // REQUIRED
     umemVirtual_ctor(&ctx->super, &ctx->host);   // REQUIRED
     ctx->super.vptr = &vtbl;                     // REQUIRED
     ctx->super.type = umemMyMemDevice;           // REQUIRED
     // Initialize device specific members:
     ...                                          // OPTIONAL
   }

The :func:`umemMyMem_ctor` function must be implemented in
:file:`umem_mymem.c` and exposed as extern function in :file:`umem.h`:

.. code-block:: c

   UMEM_EXTERN void umemMyMem_ctor(umemMyMem * const ctx,
                                   /* device specific parameters */ ...
                                   );
   
In initializing the :data:`vtbl` methods table, one can use the
default implementations for methods like :func:`calloc`,
:func:`aligned_alloc`, :func:`aligned_origin`, :func:`free` which are
provided in :file:`umem.h` and start with the prefix
:func:`umemVirtual_`. If device driver provides the corresponding
methods, their usage is highly recommended.

One must provide the implementations to the following device specific
methods: :func:`dtor`, :func:`alloc`, :func:`free`, :func:`copy_to`,
:func:`copy_from`, for instance, in :file:`umem_mymem.c` file.

Including :file:`umem_mymem.c` to CMake configuration
'''''''''''''''''''''''''''''''''''''''''''''''''''''

Update :file:`c/CMakeLists.txt` as follows:

::

   ...
   option(ENABLE_MYMEM "Enable MyMem memory context" ON)
   ...
   if (ENABLE_MYMEM)
     add_definitions("-DHAVE_MYMEM_CONTEXT")
     set(UMEM_SOURCES ${UMEM_SOURCES} umem_mymem.c)
     set(UMEM_INCLUDE_DIRS ${UMEM_INCLUDE_DIRS} <paths to MyMem include directories>)  # OPTIONAL
     set(UMEM_LIBRARIES ${UMEM_LIBRARIES} <MyMem external libraries>)                  # OPTIONAL
   endif(ENABLE_MYMEM)
   ...

Update :file:`doc/libumem/c-api.rst`
''''''''''''''''''''''''''''''''''''

Add ``:type:`umemMyMem` context`` section to "Supported storage devices" section above.

