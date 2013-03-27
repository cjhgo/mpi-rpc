# MPI-RPC

A simple, header-only library to make working with MPI more awesome.

It's integrated with the GNU pth user level threading library which
greatly simplifies certain operations.

## Requires

* GNU pth (libpth-dev)
* Boost

## Usage


### Normal MPI operations
    rpc = MPIRPC()

    // Send a message all other workers (a la broadcast)
    rpc.send_all(MessageId, some_struct);
    rpc.recv_pod(MessageId, &some_other_struct);

### Helper functions for splitting data
    // Split a message into chunks of (bufsize / num\_workers) bytes
    // and send to everyone
    vector<float> buffer(1000000);
    rpc.send_sharded(MessageId, buffer.data(), buffer.size());

### Easy grouping of workers
    // Designate a range of workers to operate on
    rpc2 = MPIRPC(0, 10)
    rpc2.send_all(...)

### Using threads

We can use the threading support to get a nicer, event-driven (RPC-like)
interface for running things like a server:

    void process_update(int worker) {
      struct WorkerUpdate w_up;
      rpc.recv_pod(worker, WorkerUpdateId + worker, &w_up);
      rpc.recv_data(worker, WorkerUpdateId + worker, &big_data, big_data_size);
    }

    for (int i = 0; i < num_workers; ++i) {
      fiber::run_forever(boost::bind(&process_update, worker));
    }

    while (1) {
      fiber::yield();
    }

Since MPI-RPC integrates with the GNU pth library, we can issue what
appear to be blocking operations; internally they are converted to the
appropriate ISend/IRecv and polled using MPI\_Test.  In the above example
we can receive updates from one or more workers in parallel, with a 
fraction of the boilerplate that we'd have to use otherwise.

## Questions/comments/bugs?

File an issue or [email me](mailto:power@cs.nyu.edu).
