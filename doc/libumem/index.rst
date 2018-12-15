.. meta::
   :robots: index, follow
   :description: libumem documentation
   :keywords: libumem, C, C++, memory management, device context

.. sectionauthor:: Pearu Peterson <pearu.peterson at quansight.com>


libumem
-------

libumem implements an abstraction for managing memory of a variety of
storage devices in an unified manner. The core part of libumem is
implemented in C for maximum portability but APIs are provided to
other programming languages such as C++ that are often easier to use
and provide better resource handling.

According to umem memory management abstraction, the data location is
described by *data address* in the given *device context*.  At the
libumem C level, the data address is given as `uintptr_t` value and
the device context is represented via C struct object that holds
various memory managment methods such as allocation, dealloction,
copying, etc. While these methods are specific to each storage device,
there is a uniquely defined interface for all supported storage
devices.  In addition, methods are provided for keeping the memory
areas of different storage devices in sync.


.. toctree::

   c-api.rst
   cxx-api.rst
