#ifndef UMEM_HPP
#define UMEM_HPP
/*
  Author: Pearu Peterson
  Created: December 2018
*/

#include <memory>
#include <iostream>
#include <string>

namespace umem {

  typedef unsigned short alignment_t;
  
  class Address;
  
  class Context {
  public:
    
    inline Address alloc(size_t nbytes);
    inline Address calloc(size_t nmemb, size_t size);
    inline Address aligned_alloc(alignment_t alignment, size_t size);


    inline bool operator == (Context &other) { return umem_is_same_device(get_raw_context(), other.get_raw_context()); }
    inline bool operator != (Context &other) { return !(*this == other); }
    
    inline bool is_ok() { return umem_is_ok(get_raw_context()); };
    inline std::string get_message() { return umem_get_message(get_raw_context()); }
    inline umemStatusType get_status() { return umem_get_status(get_raw_context()); }
    inline void set_status(umemStatusType status, const char* message) { umem_set_status(get_raw_context(), status, message); }
    inline void set_status(umemStatusType status, std::string message) { umem_set_status(get_raw_context(), status, message.c_str()); }
    inline void clear_status() { umem_clear_status(get_raw_context()); }
    
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


  class File : public Context {
  private:
    umemFile ctx;
  public:
    File(std::string filename, std::string mode="wb") {
      std::cout << "  File::File(\""<<filename<<"\", mode=\""<<mode<<"\")\n";
      umemFile_ctor(&ctx, filename.c_str(), mode.c_str());
    }
    ~File() {
      std::cout << "  File::~File()\n";
      umem_dtor(&ctx);
    }
    inline void * const get_raw_context() { return &(this->ctx); }
  };

  
  class Address {
  private:
    uintptr_t adr;
    alignment_t alignment;
    size_t nbytes;
    void * raw_ctx;
    bool own;
  public:
    Address(uintptr_t adr, alignment_t alignment,  size_t nbytes, void * raw_ctx, bool own = false):
      adr(adr), alignment(alignment), nbytes(nbytes), raw_ctx(raw_ctx), own(own) {
      std::cout << "  Address::Address("<<adr<<", "<<alignment<<", "<<nbytes<<", ctx="<<raw_ctx<<", own="<<own<<")\n";
    }
    ~Address() {
      std::cout << "  Address::~Address()["<<adr<<", "<<alignment<<", "<<nbytes<<", ctx="<<raw_ctx<<", own="<<own<<"]\n";
      if (own) {
        if (alignment)
          umem_aligned_free(raw_ctx, adr);
        else
          umem_free(raw_ctx, adr);
      }
    }

    inline void copy_to(Address& dest, size_t nbytes) { umem_copy_to(raw_ctx, adr, dest.get_raw_context(), (uintptr_t)dest, nbytes); }
    inline void copy_from(Address& src, size_t nbytes) { umem_copy_from(raw_ctx, adr, src.get_raw_context(), (uintptr_t)src, nbytes); }
    inline void copy_from(std::string src, size_t nbytes) { umem_copy_from(raw_ctx, adr, ((umemVirtual*)raw_ctx)->host, (uintptr_t)src.c_str(), nbytes); }
    inline Address connect(Context& dest, size_t dest_nbytes, size_t dest_alignment = 0) {
      return Address(umem_connect(raw_ctx, adr, dest_nbytes, dest.get_raw_context(), dest_alignment), dest_alignment, dest_nbytes, dest.get_raw_context());
    }
    inline void disconnect(Address& dest) { umem_disconnect(raw_ctx, adr, dest.get_raw_context(), (uintptr_t)dest, (alignment_t)dest); }
    inline void sync_to(Address& dest, size_t nbytes) { umem_sync_to(raw_ctx, adr, dest.get_raw_context(), (uintptr_t)dest, nbytes); }
    inline void sync_from(Address& src, size_t nbytes) { umem_sync_from(raw_ctx, adr, src.get_raw_context(), (uintptr_t)src, nbytes); }
    
    inline char& operator[] (size_t n) { return ((char*)adr)[n]; }
    inline void set(int c, size_t nbytes) { umem_set(raw_ctx, adr, c, nbytes); };

    inline operator uintptr_t() { return adr; }
    inline operator alignment_t() { return alignment; }
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
    inline bool operator == (Address& other) const { return adr == (uintptr_t)other; }
    inline bool operator == (uintptr_t i) const { return adr == i; }
    inline bool operator == (int i) const { return adr == (unsigned int)i; }
    //inline bool operator == (size_t i) const { return adr == i; }
    //inline bool operator != (uintptr_t i) const { return adr != i; }
    inline Address operator + (int i) const { return Address(adr + i, alignment, (i<(int)nbytes ? nbytes - i : 0), raw_ctx); }
    inline Address operator + (size_t i) const { return Address(adr + i, alignment, (i<nbytes ? nbytes - i : 0), raw_ctx); }
    //inline Address operator - (int i) const { return Address(adr - i, alignment, nbytes + i, raw_ctx); }
    //inline Address operator - (size_t i) const { return Address(adr - i, alignment, nbytes + i, raw_ctx); }
    inline int operator % (int i) const { return adr % i; }
    inline size_t operator % (size_t i) const { return adr % i; }
    inline Address origin() const { return (alignment ? Address(umem_aligned_origin(raw_ctx, adr), 0, nbytes, raw_ctx) : Address(adr, alignment, nbytes, raw_ctx)); }
    
  protected:
    inline void * const get_raw_context() { return raw_ctx; };
  private:
    Address(const Address&);            // Disallow copy constructor.
    Address& operator=(const Address&); // Disallow assignment operator.
  };

  // Implementations:
  inline Address Context::alloc(size_t nbytes) { return Address(umem_alloc(get_raw_context(), nbytes), 0, nbytes, get_raw_context(), true); }
  inline Address Context::calloc(size_t nmemb, size_t size) {
    return Address(umem_calloc(get_raw_context(), nmemb, size), 0, nmemb * size, get_raw_context(), true); } // TODO: check overflow
  inline Address Context::aligned_alloc(alignment_t alignment, size_t size) {
    return Address(umem_aligned_alloc(get_raw_context(), alignment, size), alignment, size, get_raw_context(), true); }
  
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
