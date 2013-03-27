#ifndef MPIRPC_H_
#define MPIRPC_H_

#include <mpi.h>
#include <vector>
#include <map>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <pth.h>

#define PANIC(...)\
  fprintf(stderr, __VA_ARGS__);\
  fprintf(stderr, "\n");\
  abort();

#define ASSERT(condition, ...)\
  if (!condition) {\
    PANIC(__VA_ARGS__);\
  }

typedef boost::function<void(void)> VoidFn;

template<class T>
struct MPITypeMapper {
  static MPI::Datatype map() {
    PANIC("Unknown type");
    return MPI::BYTE;
  }
};


struct MPIRPC {
  int _first, _last;
  MPI::Intracomm _world;

  MPIRPC() :
    _world(MPI::COMM_WORLD) {
      _first = 0;
      _last = _world.Get_size();
  }

  MPIRPC(int firstId, int lastId) :
          _first(firstId), _last(lastId), _world(MPI::COMM_WORLD) {
  }

  template<class T>
  void send_sharded(int tag, const T* data, int numElements);

  template<class T>
  void send_pod(int dst, int tag, const T& data);

  template<class T>
  void send_data(int dst, int tag, const T* data, int numElements);

  template<class T>
  void send_all(int tag, const T& data);

  void send_all(int tag, void* data, int numBytes);

  template<class T>
  pth_t fiber_send_data(int dst, int tag, const T* data, int numElements) {
    return fiber::run(
             boost::bind(&MPIRPC::send_data, this, dst, tag, data, numElements));
  }


  template<class T>
  void recv_data(int src, int tag, T* data, int numElements);

  template<class T>
  void recv_pod(int src, int tag, T* val);

  template<class T>
  void recv_all(int tag, std::vector<T>* res);

  template<class T>
  void recv_sharded(int tag, T* data, int numElements);

  template<class T>
  bool maybe_recv(int src, int tag, T* data, MPI::Status& st);
};

namespace fiber {
  static inline void* _boost_helper(void* bound_fn) {
    VoidFn* boost_fn = (VoidFn*) bound_fn;
    (*boost_fn)();
    delete boost_fn;
    return NULL;
  }
  
  // Run this function in a while(1) loop.
  static inline void _forever(VoidFn f) {
    while (1) { f(); pth_yield(NULL); }
  }

  static inline pth_t run(VoidFn f) {
    void* heap_fn = new VoidFn(f);
    pth_attr_t t_attr = pth_attr_new();
    pth_attr_init(t_attr);
    return pth_spawn(t_attr, _boost_helper, heap_fn);
  }
  
  static inline void run_forever(VoidFn f) {
    run(boost::bind(&_forever, f));
  }

  static inline void wait(std::vector<pth_t>& fibers) {
    for (int i = 0; i < fibers.size(); ++i) {
      ASSERT(pth_join(fibers[i], NULL), "failed to join with fiber.");
    }

    fibers.clear();
  }

  static inline void yield() {
    pth_yield(NULL);
  }
}

template<>
struct MPITypeMapper<void> {
  static MPI::Datatype map() {
    return MPI::BYTE;
  }
};

template<>
struct MPITypeMapper<char> {
  static MPI::Datatype map() {
    return MPI::BYTE;
  }
};

template<>
struct MPITypeMapper<float> {
  static MPI::Datatype map() {
    return MPI::FLOAT;
  }
};

template<>
struct MPITypeMapper<double> {
  static MPI::Datatype map() {
    return MPI::DOUBLE;
  }
};
template<class T>
inline void MPIRPC::send_pod(int dst, int tag, const T& data) {
  send_data(dst, tag, (char*) (&data), sizeof(T));
}

template<class T>
inline void MPIRPC::send_data(int dst, int tag, const T* data, int numElements) {
  MPI::Datatype type = MPITypeMapper<T>::map();
  MPI::Request pending = _world.Isend(data, numElements, type, dst, tag);
  MPI::Status status;
  while (!pending.Test(status)) {
    fiber::yield();
  }
  ASSERT(status.Get_count(type) == numElements,
       "Send did not match recv: %d vs %d", numElements, status.Get_count(type));
}

inline void MPIRPC::send_all(int tag, void* data, int numBytes) {
  for (int i = _first; i <= _last; ++i) {
    send_data(i, tag, data, numBytes);
  }
}

template<class T>
inline void MPIRPC::send_all(int tag, const T& data) {
  for (int i = _first; i <= _last; ++i) {
    send_pod(i, tag, data);
  }
}

template<class T>
inline void MPIRPC::send_sharded(int tag, const T* data, int numElements) {
  int64_t numServers = (_last - _first + 1);
  int64_t elemsPerServer = numElements / numServers;
  for (int j = 0; j < numServers; ++j) {
    int64_t offset = j * elemsPerServer;
    int64_t nextOffset = (j == numServers - 1) ? numElements : (j + 1) * elemsPerServer;
    int dst = _first + j;

//    fprintf(stderr, "%p %d %d %d %d\n", data, offset, nextOffset, nextOffset - offset, numElements);
    send_data(dst, tag, data + offset, nextOffset - offset);
  }
}

template<class T>
inline void MPIRPC::recv_data(int src, int tag, T* data, int numElements) {
  MPI::Status status;
  MPI::Datatype type = MPITypeMapper<T>::map();
  while (!_world.Iprobe(src, tag, status)) {
    fiber::yield();
  }

  ASSERT(status.Get_count(type) == numElements,
         "Receive did not match send: %d vs %d", numElements, status.Get_count(type));
  MPI::Request pending = _world.Irecv(data, numElements, type, src, tag);
  while (!pending.Test(status)) {
    fiber::yield();
  }

  ASSERT(status.Get_count(type) == numElements,
       "Receive did not match send: %d vs %d", numElements, status.Get_count(type));
}

template<class T>
inline void MPIRPC::recv_pod(int src, int tag, T* val) {
  recv_data(src, tag, (char*) (val), sizeof(T));
}

template<class T>
inline void MPIRPC::recv_all(int tag, std::vector<T>* res) {
  res->resize(_last - _first + 1);
  for (int i = _first; i <= _last; ++i) {
    recv_pod(i, tag, &res->at(i - _first));
  }
}
template<class T>
inline void MPIRPC::recv_sharded(int tag, T* data, int numElements) {
  int64_t numServers = (_last - _first + 1);
  int64_t elemsPerServer = numElements / numServers;
  for (int j = 0; j < numServers; ++j) {
    int64_t offset = j * elemsPerServer;
    int64_t nextOffset = (j == numServers - 1) ? numElements : (j + 1) * elemsPerServer;
    int src = _first + j;
    recv_data(src, tag, data + offset, nextOffset - offset);
  }
}
#endif /* MPIRPC_H_ */
