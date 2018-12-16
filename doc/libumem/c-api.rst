.. meta::
   :robots: index,follow
   :description: libumem documentation

.. sectionauthor:: Pearu Peterson <pearu.peterson at quansight.com>

libumem C API
=============

Memory location
---------------

Within libumem C-API, the data location address is a `uintptr_t`
value. In the case of host RAM, the address value is equal to data
pointer value. For other storage devices, the address value may have
various interpretations that depends on the storage device as well as
the storage device driver library. However, the fundamental assumption
of address value is that its increments give valid addresses of the
whole data content stored in the device.


Supported storage devices
-------------------------

The libumem C-API provides the following device memory context
objects (C `struct` instances):

* `umemHost` - `stdlib.h` based interface to host RAM,

* `umemFile` - `stdio.h` based interface to files,

* `umemCuda` - CUDA based interface to GPU device memory.

Each device memory context has specific initializer (a
constructor). However, all other memory management methods are
universal among the supported storage devices.

`umemHost`
''''''''''

The `umemHost` C `struct` is a host RAM context and it must be
initialized using the constructor function :func:`umemHost_ctor`:

.. code-block:: c

   void umemHost_ctor(umemHost * const this);

To destruct the host RAM context object, use :func:`umem_dtor`
destructor function. See below.

`memFile`
'''''''''

The `umemFile` C `struct` is a file context that must be initialized
with the following constructor function:

.. code-block:: c

   void umemFile_ctor(umemFile * const ctx, const char * filename, const char * mode);

Here *filename* is the path name of a file that is opened using given
*mode*. The *mode* string must start with one of the following
strings: `r`, `r+`, `w`, `w+`, `a`, `a+`. The `*mode*` string may
include also the character `b` to indicate binary file content.

The destructor function :func:`umem_dtor` closes the file.


`memCuda`
'''''''''

The `umemCuda` C `struct` is a CUDA based GPU device memory context
that must be initialized with the following constructor function:

.. code-block:: c

   void umemCuda_ctor(umemCuda * const ctx, int device);

Here *device* is GPU device number. The constructor function will set
the corresponding GPU device.

While the destructor function :func:`umem_dtor` does not call any CUDA
API functions, it is recommended to use it to destruct `umemCuda`
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

Allocates *nbytes* of memory in the given storage device. The
allocated memory is uninitialized.

.. code-block:: c

   uintptr_t umem_calloc(void const * ctx, size_t nmemb, size_t size);

Allocated an array of given *size* and member byte size
*nmemb*. Returns the starting address of allocated memory. The
allocated memory is zero-initialized.

.. code-block:: c

   void umem_free(void const * ctx, uintptr_t adr);

Frees the memory that was allocated with methods
:func:`umem_alloc` or :func:`umem_calloc`.

.. code-block:: c

   uintptr_t umem_aligned_alloc(void const * ctx, size_t alignement, size_t size);

Allocates *size* bytes (plus some extra) of device memory so that the
returned starting address is aligned to given *alignement* value.

.. code-block:: c

   uintptr_t umem_free_aligned(void const * ctx, uintptr_t adr);

Frees the memory that was allocated with methods
:func:`umem_aligned_alloc`.

.. code-block:: c

   uintptr_t umem_aligned_origin(void const * ctx, uintptr_t adr);

Return the original memory address that was obtained when allocating
device memory with :func:`umem_aligned_alloc`.

Memory initialization
'''''''''''''''''''''

For initializing device memory with arbitrary data from host RAM, see
below how to copy data between devices.

.. code-block:: c

   uintptr_t umem_set(void const * ctx, uintptr_t adr, int c, size_t nbytes);

Sets *nbytes* of device memory with starting address *adr* to byte
value *c* (byte-wise).

Copying data between memory devices
'''''''''''''''''''''''''''''''''''

`umem_copy_to`, `umem_copy_from`, `umem_copy_to_safe`, `umem_copy_from_safe`

Keeping data in sync between memory devices
'''''''''''''''''''''''''''''''''''''''''''

`umem_connect`, `umem_disconnect`, `umem_sync_to`, `umem_sync_from`, `umem_sync_to_safe`, `umem_sync_from_safe`

Error message handling
----------------------

`umem_get_status`, `umem_get_message`, `umem_set_status`, `umem_clean_status`, `umem_is_ok`

Utility functions
-----------------

`umem_is_same_device`, `umem_get_device_name`
