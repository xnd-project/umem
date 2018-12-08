#ifndef UMEM_HPP
#define UMEM_HPP
/*
  Author: Pearu Peterson
  Created: December 2018
*/

#include <memory>
#include <iostream>

namespace umem {

  class Address;
  
  class Context {
  public:
    
    inline Address alloc(size_t nbytes);
    inline Address calloc(size_t nmemb, size_t size);
    inline Address aligned_alloc(size_t alignment, size_t size);

    inline bool is_ok() { return umem_is_ok(get_raw_context()); };
    inline bool operator == (Context &other) { return umem_is_same_device(get_raw_context(), other.get_raw_context()); }
    
    virtual void * const get_raw_context() = 0;
  };


  class Host : public Context {
  private:
    umemHost ctx;
  public:
    Host() {
      std::cout << "  Host::Host()\n";
      umemHost_ctor(&ctx);
    }
    ~Host() {
      std::cout << "  Host::~Host()\n";
      umem_dtor(&ctx);
    }

    inline void * const get_raw_context() { return &(this->ctx); }
  };

  
  class Address {
  public:
    uintptr_t adr;
    size_t alignment;
  private:
    void * raw_ctx;
    bool own;
  public:
    Address(uintptr_t adr, size_t alignment, void * raw_ctx, bool own = false):
      adr(adr), alignment(alignment), raw_ctx(raw_ctx), own(own) {
      std::cout << "  Address::Address("<<adr<<", "<<alignment<<", "<<raw_ctx<<", own="<<own<<")\n";
    }
    ~Address() {
      std::cout << "  Address::~Address()[own="<<own<<", "<<adr<<", ctx="<<raw_ctx<<"]\n";
      if (own) {
        if (alignment)
          umem_aligned_free(raw_ctx, adr);
        else
          umem_free(raw_ctx, adr);
      }
    }

    inline void copy_to(Address& dest, size_t nbytes) { umem_copy_to(raw_ctx, adr, dest.get_raw_context(), dest.adr, nbytes); }
    inline void copy_from(Address& src, size_t nbytes) { umem_copy_to(raw_ctx, adr, src.get_raw_context(), src.adr, nbytes); }
    inline Address connect(size_t nbytes, Context& dest, size_t alignment) {
      return Address(umem_connect(raw_ctx, adr, nbytes, dest.get_raw_context(), alignment), alignment, dest.get_raw_context());
    }
    inline void disconnect(Address& dest) { umem_disconnect(raw_ctx, adr, dest.get_raw_context(), dest.adr, dest.alignment); }
    inline void sync_to(Address& dest, size_t nbytes) { umem_sync_to(raw_ctx, adr, dest.get_raw_context(), dest.adr, nbytes); }
    inline void sync_from(Address& src, size_t nbytes) { umem_sync_to(raw_ctx, adr, src.get_raw_context(), src.adr, nbytes); }
    
    inline char& operator[] (size_t n) { return ((char*)adr)[n]; }
    inline void set(int c, size_t nbytes) { umem_set(raw_ctx, adr, c, nbytes); };
    inline operator void*() { return (void*)adr; }
    inline operator bool*() { return (bool*)adr; }
    inline operator char*() { return (char*)adr; }
    inline operator short*() { return (short*)adr; }
    inline operator int*() { return (int*)adr; }
    inline operator long*() { return (long*)adr; }
    inline operator long long*() { return (long long*)adr; }
    inline operator float*() { return (float*)adr; }
    inline operator double*() { return (double*)adr; }
    inline operator unsigned char*() { return (unsigned char*)adr; }
    inline operator unsigned short*() { return (unsigned short*)adr; }
    inline operator unsigned int*() { return (unsigned int*)adr; }
    inline operator unsigned long*() { return (unsigned long*)adr; }
    inline operator unsigned long long*() { return (unsigned long long*)adr; }
    /* TODO: add support to other casts as well. To support casting to
       the pointers of user-defined types, this could be done as follows:

         // Inside user code:
         #define USER_DEFINED_UMEM_CASTS \
           inline operator mytype*() { return (mytype*)adr; } \
           inline operator anothertype*() { return (anothertype*)adr; }
         #include "umem.h"

       and inside Address class definition have

         #ifdef USER_DEFINED_UMEM_CASTS
         USER_DEFINED_UMEM_CASTS
         #endif
     */

    inline Address operator + (uintptr_t i) const { return Address(adr + i, alignment, raw_ctx); }
    inline Address operator - (uintptr_t i) const { return Address(adr - i, alignment, raw_ctx); }

  protected:
    inline void * const get_raw_context() { return raw_ctx; };
  };

  // Implementations:
  inline Address Context::alloc(size_t nbytes) { return Address(umem_alloc(get_raw_context(), nbytes), 0, get_raw_context()); }
  inline Address Context::calloc(size_t nmemb, size_t size) {
    return Address(umem_calloc(get_raw_context(), nmemb, size), 0, get_raw_context()); }
  inline Address Context::aligned_alloc(size_t alignment, size_t size) {
    return Address(umem_aligned_alloc(get_raw_context(), alignment, size), alignment, get_raw_context()); }
  
  class Connection {
  public:
    Connection(std::shared_ptr<Context*const> &src, 
               std::shared_ptr<Context*const> &dest): src_(src), dest_(dest) {}
    
  private:
    std::shared_ptr<Context*const> &src_;
    std::shared_ptr<Context*const> &dest_;
  };
}

#endif
