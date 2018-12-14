#ifndef UMEM_HPP
#define UMEM_HPP
/**

   C++ API of umem library

   @author: Pearu Peterson
   Created: December 2018
*/

#include <memory>
#include <iostream>
#include <string>

namespace umem {

  class Address;

  class Context {
  public:
    /**
       Allocate context memory

       @param[in]    nbytes - the size of allocated memory in bytes.
       @return       address of allocated memory
     */

    Address alloc(size_t nbytes);
    /**
       Allocate context memory as zero initialized array.

       @param[in]    nmemb - number of array members
       @param[in]    size - size of array member in bytes
       @return       address of allocated memory
     */

    Address calloc(size_t nmemb, size_t size);
    /**
       Allocate context memory with alignment

       @param[in]    alignment - required alignement
       @param[in]    size - the size of allocated memory in bytes. The size must be a multiple of alignment.
       @return       address of allocated memory
     */
    Address aligned_alloc(size_t alignment, size_t size);

    /**
       Test device context equality.
     */
    bool operator == (Context &other);
    bool operator != (Context &other);

    /**
       Test the status of the device context.

       @return       true if the status is OK
     */
    bool is_ok();

    /**
       Return current message from device context.
       
       @return       message, empty string if none
     */
    std::string get_message();

    /**
       Return status of the device context.

       @return       status value
     */
    umemStatusType get_status();

    /**
       Set device context status with message

       @param[in]    status
       @param[in]    message
     */
    void set_status(umemStatusType status, const char* message);
    void set_status(umemStatusType status, std::string message);

    /**
       Clears device context status.
     */
    void clear_status();

    /// Internal methods:

    /**
       Return raw pointer of context structure. Used internally.

       @returns      pointer
     */
    virtual void * const get_raw_context() = 0;
  protected:
    /// Constructing base class object is disallowed.
    Context() {}
  };

  /**
     Host memory context represents RAM
   */
  class Host : public Context {
  public:
    /**
       Constructor of a host memory.

       @see Context() for available API methods.
     */    
    Host();

    /// Internal methods and members
    ~Host();
    void * const get_raw_context();
  private:
    umemHost ctx;
  };

  /**
     File memory context represents a file storage.

     The File context is provided mostly for demonstration of
     umem. For efficient mapping to a file use Mmap memory context
     [NOT IMPLEMENTED].
   */
  class File : public Context {

  public:
    /**
       Constructor of a file memory.
 
       @param[in]    filename - specify file name
       @param[in]    mode - specify mode for opening the file
     */
    File(std::string filename, std::string mode="wb");

    /// Internal methods and members
    ~File();
    void * const get_raw_context();
  private:
    umemFile ctx;
  };

  /**
     Address represents a memory address in an arbitrary storage
     device.

   */
  class Address {
  public:
    /**
       Constructor of Address object. Used internally.

       @param[in]     adr - device memory address value
       @param[in]     alignment - memory alignment value
       @param[in]     size - size of memory extent in bytes
       @param[in]     raw_ctx - raw pointer to internal context object
       @param[in]     own - true if Address destructor must handle freeing the memory
       @param[in]     mate - pointer to connected Address object
     */    
    Address(uintptr_t adr, size_t alignment,  size_t size,
            void * raw_ctx, bool own = false, Address* mate = NULL);

    /**
       Return the extent of interfaced memory in bytes.
    */
    size_t size() const;

    /**
       Return the alignment used to allocate the memory.
     */
    size_t alignment() const;

    /**
       Copy data from given address to another address of a possibly different context 

       @param[out]    dest - destination Address object
       @param[in]     nbytes - size of data to be copied
     */
    void copy_to(Address& dest, size_t nbytes);

    /**
       Copy data to given address from another address of a possibly different context

       @param[in]     src - source Address object
       @param[in]     nbytes - size of data to be copied
     */
    void copy_from(Address& src, size_t nbytes);

    /**
       Copy data to given address from a string (host context).

       @param[in]     src - string object in host
       @param[in]     nbytes - size of data to be copied
     */
    void copy_from(std::string src, size_t nbytes);

    /**
       Create a connection to another device context as a buffer.

       @param[in]     dest_ctx - context of another device
       @param[in]     dest_nbytes - size of memory extent to be paired
       @param[in]     dest_alignment - required alignement of the paired address. 0 means default alignment.
       @return        address - paired address with dest_nbytes size
     */
    
    Address connect(Context& dest_ctx, size_t dest_nbytes, size_t dest_alignment = 0);

    /**
       Syncronize data from given address to destination address.

       @param[out]    dest - destination address, should be paired with given address.
       @param[in]     nbytes - size of data to be synchronized
     */
    void sync_to(Address& dest, size_t nbytes);

    /**
       Syncronize data to given address from the source address.

       @param[in]     src - source address, should be paired with given address.
       @param[in]     nbytes - size of data to be synchronized
     */
    void sync_from(Address& src, size_t nbytes);

    /**
       Syncronize data to a paired address.

       @param[in]     nbytes - size of data to be synchronized
     */
    void sync(size_t nbytes);

    /**
       Update data from a paired address.

       @param[in]     nbytes - size of data to be synchronized
     */
    void update(size_t nbytes);

    /**
       Access host address data as char buffer. A convenience method.

       @param[in]     n - byte index of data
     */
    char& operator[] (size_t n);

    /**
       Fill memory with a given byte value.

       @param[in]     c - byte value
       @param[in]     nbytes - number of bytes to be filled
     */
    void set(int c, size_t nbytes);

    /**
       Obtain address value of device memory.

       @return        adr - address value
     */
    operator uintptr_t();

    /**
       Standard casting operators. The address must be in host context.

       To cast to an user-defined type `mytype`, use

         mytype* ptr = (mytype*)(void*)adr

       where adr is Address object.
     */
    ///TODO: check for host context
    inline operator void*() { return (void*)adr_; }
    inline operator bool*() { return (bool*)adr_; }
    inline operator char*() { return (char*)adr_; }
    inline operator short*() { return (short*)adr_; }
    inline operator int*() { return (int*)adr_; }
    inline operator long*() { return (long*)adr_; }
    inline operator long long*() { return (long long*)adr_; }
    inline operator float*() { return (float*)adr_; }
    inline operator double*() { return (double*)adr_; }
    inline operator unsigned char*() { return (unsigned char*)adr_; }
    inline operator unsigned short*() { return (unsigned short*)adr_; }
    inline operator unsigned int*() { return (unsigned int*)adr_; }
    inline operator unsigned long*() { return (unsigned long*)adr_; }
    inline operator unsigned long long*() { return (unsigned long long*)adr_; }

    /**
      Equality operators.
     */
    bool operator == (Address& other) const;
    bool operator == (uintptr_t i) const;
    bool operator == (int i) const;

    /**
      Pointers artihmetic support.
    */
    Address operator + (int i) const;
    Address operator + (size_t i) const;
    int operator % (int i) const;
    size_t operator % (size_t i) const;

    /// Internal methods and members:
    ~Address();
    
    /**
      Return the memory address of allocation origin.
    */
    Address origin() const;

  protected:

    /**
       Return pointer to internal context object.
    */
    inline void * const get_raw_context() { return raw_ctx; };
  private:
    uintptr_t adr_;      /// memory address value
    size_t alignment_;   /// alignment of memory address
    size_t size_;        /// size of memory extent
    void * raw_ctx;      /// pointer to internal context
    bool own;            /// if true then memory is freed in Address destructor
    Address* mate_;      /// paired address

    Address& operator=(const Address&); // Disallow assignment operator.
    //Address(const Address&);          // Disallow copy constructor. MSVC requires copy constructor.
  };

  /**

     Implementations of various class methods. Read only if you really want to.
     
  */
  
  // Context implementations:
  inline Address Context::alloc(size_t nbytes) { return Address(umem_alloc(get_raw_context(), nbytes), 0, nbytes, get_raw_context(), true); }
  inline Address Context::calloc(size_t nmemb, size_t size) {
    return Address(umem_calloc(get_raw_context(), nmemb, size), 0, nmemb * size, get_raw_context(), true); } // TODO: check overflow
  inline Address Context::aligned_alloc(size_t alignment, size_t size) {
    return Address(umem_aligned_alloc(get_raw_context(), alignment, size), alignment, size, get_raw_context(), true); }
  inline bool Context::operator == (Context &other) { return umem_is_same_device(get_raw_context(), other.get_raw_context()); }
  inline bool Context::operator != (Context &other) { return !(*this == other); }
  inline bool Context::is_ok() { return umem_is_ok(get_raw_context()); }
  inline std::string Context::get_message() { return umem_get_message(get_raw_context()); }
  inline umemStatusType Context::get_status() { return umem_get_status(get_raw_context()); }
  inline void Context::set_status(umemStatusType status, const char* message) { umem_set_status(get_raw_context(), status, message); }
  inline void Context::set_status(umemStatusType status, std::string message) { umem_set_status(get_raw_context(), status, message.c_str()); }
  inline void Context::clear_status() { umem_clear_status(get_raw_context()); }

  // Host implementations:

  Host::Host() { umemHost_ctor(&ctx); }
  Host::~Host() { umem_dtor(&ctx); }
  inline void * const Host::get_raw_context() { return &(this->ctx); }
  
  // File implementations:

  File::File(std::string filename, std::string mode) {
    umemFile_ctor(&ctx, filename.c_str(), mode.c_str());
  }
  File::~File() { umem_dtor(&ctx); }
  inline void * const File::get_raw_context() { return &(this->ctx); }
  
  // Cuda implementations:

  // Address implementations

  Address::Address(uintptr_t adr, size_t alignment,  size_t size,
                   void * raw_ctx, bool own, Address* mate):
    adr_(adr), alignment_(alignment), size_(size), raw_ctx(raw_ctx), own(own), mate_(mate) {}
  
  Address::~Address() {
    if (own) {
      if (mate_ == NULL) {
        if (alignment_)
          umem_aligned_free(raw_ctx, adr_);
        else
          umem_free(raw_ctx, adr_);
      } else
        umem_disconnect(mate_->get_raw_context(), (uintptr_t)(*mate_), raw_ctx, (uintptr_t)adr_, alignment_);
    }
  }

  inline size_t Address::size() const { return size_; }
  inline size_t Address::alignment() const { return alignment_; }
  inline void Address::copy_to(Address& dest, size_t nbytes) { umem_copy_to_safe(raw_ctx, adr_, size_, dest.get_raw_context(), (uintptr_t)dest, dest.size(), nbytes); }
  inline void Address::copy_from(Address& src, size_t nbytes) { umem_copy_from_safe(raw_ctx, adr_, size_, src.get_raw_context(), (uintptr_t)src, src.size(), nbytes); }
  inline void Address::copy_from(std::string src, size_t nbytes) { umem_copy_from_safe(raw_ctx, adr_, size_, ((umemVirtual*)raw_ctx)->host, (uintptr_t)src.c_str(), src.size(), nbytes); }
  inline Address Address::connect(Context& dest_ctx, size_t dest_nbytes, size_t dest_alignment) {
    return Address(umem_connect(raw_ctx, adr_, dest_nbytes, dest_ctx.get_raw_context(), dest_alignment), dest_alignment, dest_nbytes, dest_ctx.get_raw_context(), true, this);
  }
  inline void Address::sync_to(Address& dest, size_t nbytes) { umem_sync_to_safe(raw_ctx, adr_, size_, dest.get_raw_context(), (uintptr_t)dest, dest.size(), nbytes); }
  inline void Address::sync_from(Address& src, size_t nbytes) { umem_sync_from_safe(raw_ctx, adr_, size_, src.get_raw_context(), (uintptr_t)src, src.size(), nbytes); }
  inline void Address::sync(size_t nbytes) { assert(mate_ != NULL); umem_sync_to_safe(raw_ctx, adr_, size_, mate_->get_raw_context(), (uintptr_t)(*mate_), mate_->size(), nbytes); }
  inline void Address::update(size_t nbytes) { assert(mate_ != NULL);umem_sync_from_safe(raw_ctx, adr_, size_, mate_->get_raw_context(), (uintptr_t)(*mate_), mate_->size(), nbytes); }
  inline void Address::set(int c, size_t nbytes) { umem_set_safe(raw_ctx, adr_, size_, c, nbytes); }
  inline char& Address::operator[] (size_t n) { assert(n<size_); return ((char*)adr_)[n]; } // TODO: ensure that ctx is a host context
  inline Address::operator uintptr_t() { return adr_; }

  inline bool Address::operator == (Address& other) const { return adr_ == (uintptr_t)other; }
  inline bool Address::operator == (uintptr_t i) const { return adr_ == i; }
  inline bool Address::operator == (int i) const { return adr_ == (unsigned int)i; }

  inline Address Address::operator + (int i) const { return Address(adr_ + i, alignment_, (i<(int)size_ ? size_ - i : 0), raw_ctx); }
  inline Address Address::operator + (size_t i) const { return Address(adr_ + i, alignment_, (i<size_ ? size_ - i : 0), raw_ctx); }
  inline int Address::operator % (int i) const { return adr_ % i; }
  inline size_t Address::operator % (size_t i) const { return adr_ % i; }
  inline Address Address::origin() const { return (alignment_ ? Address(umem_aligned_origin(raw_ctx, adr_), 0, size_, raw_ctx) : Address(adr_, alignment_, size_, raw_ctx)); }
}

#endif
