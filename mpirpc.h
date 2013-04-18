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

struct ShardCalc {
  int num_workers_;
  int num_elements_;
  int elem_size_;

  ShardCalc(int num_elements, int elem_size = 1, int num_workers = MPI::COMM_WORLD.Get_size()) :
      num_workers_(num_workers), num_elements_(num_elements), elem_size_(elem_size) {
  }

  int start(int worker) {
    int64_t elems_per_server = num_elements_ / num_workers_;
    int64_t offset = worker * elems_per_server;
    if (offset > num_elements_) {
      offset = num_elements_;
    }
    return offset * elem_size_;
  }

  int end(int worker) {
    int64_t elems_per_server = num_elements_ / num_workers_;
    int64_t offset = (worker + 1) * elems_per_server;
    if (offset > num_elements_) {
      offset = num_elements_;
    }
    return offset * elem_size_;
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
  void send_sharded(int tag, const T* data, int num_elems);

  void send_sharded(int tag, const char* data, int elem_size, int num_elems);

  template<class T>
  void send_pod(int dst, int tag, const T& data);

  template<class T>
  void send_data(int dst, int tag, const T* data, int num_elems);

  template<class T>
  void send_all(int tag, const T& data);

  void send_all(int tag, const void* data, int numBytes);

  template<class T>
  pth_t fiber_send_data(int dst, int tag, const T* data, int num_elems);

  template<class T>
  void recv_data(int src, int tag, T* data, int num_elems);

  template<class T>
  void recv_pod(int src, int tag, T* val);

  template<class T>
  void recv_all(int tag, std::vector<T>* res);

  template<class T>
  void recv_sharded(int tag, T* data, int num_elems);

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
  while (1) {
    f();
    pth_yield(NULL);
  }
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
inline void MPIRPC::send_data(int dst, int tag, const T* data, int num_elems) {
  MPI::Datatype type = MPITypeMapper<T>::map();
  MPI::Request pending = _world.Isend(data, num_elems, type, dst, tag);
  MPI::Status status;
  while (!pending.Test(status)) {
    fiber::yield();
  }
  ASSERT(status.Get_count(type) == num_elems, "Send did not match recv: %d vs %d", num_elems, status.Get_count(type));
}

inline void MPIRPC::send_all(int tag, const void* data, int numBytes) {
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
inline void MPIRPC::send_sharded(int tag, const T* data, int num_elems) {
  send_sharded(tag, data, sizeof(T), num_elems);
}

inline void MPIRPC::send_sharded(int tag, const char* data, int elem_size, int num_elems) {
  int64_t num_servers = (_last - _first + 1);
  int64_t elems_per_server = num_elems / num_servers;
  for (int j = 0; j < num_servers; ++j) {
    int64_t offset = elem_size * j * elems_per_server;
    int64_t next = elem_size * ((j == num_servers - 1) ? num_elems : (j + 1) * elems_per_server);
    int dst = _first + j;

    //    fprintf(stderr, "%p %d %d %d %d\n", data, offset, next, next - offset, num_elems);
    send_data(dst, tag, data + offset, next - offset);
  }
}

template<class T>
inline void MPIRPC::recv_data(int src, int tag, T* data, int num_elems) {
  MPI::Status status;
  MPI::Datatype type = MPITypeMapper<T>::map();
  while (!_world.Iprobe(src, tag, status)) {
    fiber::yield();
  }

  ASSERT(status.Get_count(type) == num_elems,
         "Receive did not match send: %d vs %d", num_elems, status.Get_count(type));
  MPI::Request pending = _world.Irecv((void*) data, num_elems, type, src, tag);
  while (!pending.Test(status)) {
    fiber::yield();
  }

  ASSERT(status.Get_count(type) == num_elems,
         "Receive did not match send: %d vs %d", num_elems, status.Get_count(type));
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
inline void MPIRPC::recv_sharded(int tag, T* data, int num_elems) {
  int64_t num_servers = (_last - _first + 1);
  int64_t elems_per_server = num_elems / num_servers;
  for (int j = 0; j < num_servers; ++j) {
    int64_t offset = j * elems_per_server;
    int64_t next = (j == num_servers - 1) ? num_elems : (j + 1) * elems_per_server;
    int src = _first + j;
    recv_data(src, tag, data + offset, next - offset);
  }
}

template<class T>
inline pth_t MPIRPC::fiber_send_data(int dst, int tag, const T* data, int num_elems) {
  return fiber::run(boost::bind(&MPIRPC::send_data, this, dst, tag, data, num_elems));
}

#endif /* MPIRPC_H_ */
