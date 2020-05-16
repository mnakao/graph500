#ifndef UTILS_IMPL_HPP_
#define UTILS_IMPL_HPP_
#include <stdint.h>
#include <stdarg.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <sched.h>
#include <omp.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/shm.h>
#include <algorithm>
#include <vector>
#include <deque>
#include "primitives.hpp"
#include "utils_core.h"

void print_with_prefix(const char* format, ...);
#define debug_print(prefix, ...)

struct COMM_2D {
	MPI_Comm comm;
	int rank, rank_x, rank_y;
	int size, size_x, size_y;
	int* rank_map; // Index: rank_x + rank_y * size_x
};

static void swap(COMM_2D& a, COMM_2D& b) {
	COMM_2D tmp = b;
	b = a;
	a = tmp;
}

struct MPI_GLOBALS {
	int rank;
	int size;
	int thread_level;
	int rank_2d;
	int rank_2dr;
	int rank_2dc;
	int size_2d; // = size
	int size_2dc; // = comm_2dr.size()
	int size_2dr; // = comm_2dc.size()
	MPI_Comm comm_2d;
	MPI_Comm comm_2dr; // = comm_x
	MPI_Comm comm_2dc;
	bool isRowMajor;
	COMM_2D comm_r;
	COMM_2D comm_c;
	bool isMultiDimAvailable;
	int rank_y;
	int rank_z;
	int size_y; // = comm_y.size()
	int size_z; // = comm_z.size()
	MPI_Comm comm_y;
	MPI_Comm comm_z;

	// utility method
	bool isMaster() const { return rank == 0; }
	bool isRmaster() const { return rank == size-1; }
	bool isYdimAvailable() const { return comm_y != comm_2dc; }
};

MPI_GLOBALS mpi;

//-------------------------------------------------------------//
// For generic typing
//-------------------------------------------------------------//

template <> struct MpiTypeOf<char> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<char>::type = MPI_CHAR;
template <> struct MpiTypeOf<short> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<short>::type = MPI_SHORT;
template <> struct MpiTypeOf<int> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<int>::type = MPI_INT;
template <> struct MpiTypeOf<long> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<long>::type = MPI_LONG;
template <> struct MpiTypeOf<long long> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<long long>::type = MPI_LONG_LONG;
template <> struct MpiTypeOf<float> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<float>::type = MPI_FLOAT;
template <> struct MpiTypeOf<double> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<double>::type = MPI_DOUBLE;
template <> struct MpiTypeOf<unsigned char> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<unsigned char>::type = MPI_UNSIGNED_CHAR;
template <> struct MpiTypeOf<unsigned short> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<unsigned short>::type = MPI_UNSIGNED_SHORT;
template <> struct MpiTypeOf<unsigned int> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<unsigned int>::type = MPI_UNSIGNED;
template <> struct MpiTypeOf<unsigned long> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<unsigned long>::type = MPI_UNSIGNED_LONG;
template <> struct MpiTypeOf<unsigned long long> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<unsigned long long>::type = MPI_UNSIGNED_LONG_LONG;

template <typename T> struct template_meta_helper { typedef void type; };
template <typename T> MPI_Datatype get_mpi_type(T& instance) {
	return MpiTypeOf<T>::type;
}

void throw_exception(const char* format, ...) {
	char buf[300];
	va_list arg;
	va_start(arg, format);
    vsnprintf(buf, sizeof(buf), format, arg);
    va_end(arg);
    fprintf(IMD_OUT, "[r:%d] %s\n", mpi.rank, buf);
    throw buf;
}

void print_with_prefix(const char* format, ...) {
	char buf[300];
	va_list arg;
	va_start(arg, format);
    vsnprintf(buf, sizeof(buf), format, arg);
    va_end(arg);
    fprintf(IMD_OUT, "%s\n", buf);
}

void* xMPI_Alloc_mem(size_t nbytes) {
  void* p = NULL;
  MPI_Alloc_mem(nbytes, MPI_INFO_NULL, &p);
  return p;
}

void* cache_aligned_xcalloc(const size_t size) {
    void* p = NULL;
	posix_memalign(&p, CACHE_LINE, size);
	memset(p, 0, size);
	return p;
}

void* cache_aligned_xmalloc(const size_t size) {
	void* p = NULL;
	posix_memalign(&p, CACHE_LINE, size);
	return p;
}

void* page_aligned_xcalloc(const size_t size) {
	void* p = NULL;
	posix_memalign(&p, PAGE_SIZE, size);
	memset(p, 0, size);
	return p;
}
void* page_aligned_xmalloc(const size_t size) {

	void* p = NULL;
	posix_memalign(&p, PAGE_SIZE, size);
	return p;
}

template <typename T> T roundup(T size, T width)
{
	return (size + width - 1) / width * width;
}

template <typename T>
void get_partition(T size, int num_part, int part_idx, T& begin, T& end) {
	T part_size = (size + num_part - 1) / num_part;
	begin = std::min(part_size * part_idx, size);
	end = std::min(begin + part_size, size);
}

// # of partition = num_blks * parts_factor
template <typename T>
void get_partition(T* blk_offset, int num_blks, int parts_per_blk, int part_idx, T& begin, T& end) {
	int blk_idx = part_idx / parts_per_blk;
	T blk_begin = blk_offset[blk_idx];
	T blk_size = blk_offset[blk_idx+1] - blk_begin;
	get_partition(blk_size, parts_per_blk, part_idx - blk_idx * parts_per_blk, begin, end);
	begin += blk_begin;
	end += blk_begin;
}

template <typename T>
void get_partition(int64_t size, T* sorted, int log_blk,
		int num_part, int part_idx, int64_t& begin, int64_t& end)
{
	if(size == 0) {
		begin = end = 0;
		return ;
	}
	get_partition(size, num_part, part_idx, begin, end);
	if(begin != 0) {
		T piv = sorted[begin] >> log_blk;
		while(begin < size && (sorted[begin] >> log_blk) == piv) ++begin;
	}
	if(end != 0) {
		T piv = sorted[end] >> log_blk;
		while(end < size && (sorted[end] >> log_blk) == piv) ++end;
	}
}

static void setup_2dcomm()
{
	bool success = false;
	mpi.isMultiDimAvailable = false;

	if(!success) {
		int twod_r = 1, twod_c = 1;
		const char* twod_r_str = getenv("TWOD_R");
		if(twod_r_str){
			twod_r = atoi((char*)twod_r_str);
			twod_c = mpi.size / twod_r;
			if(twod_r == 0 || (twod_c * twod_r) != mpi.size) {
				if(mpi.isMaster()) print_with_prefix("# of MPI processes(%d) cannot be divided by %d", mpi.size, twod_r);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
		else {
			for(twod_r = (int)sqrt(mpi.size); twod_r < mpi.size; ++twod_r) {
				twod_c = mpi.size / twod_r;
				if(twod_c * twod_r == mpi.size) {
					break;
				}
			}
			if(twod_c * twod_r != mpi.size) {
				if(mpi.isMaster()) print_with_prefix("Could not find the RxC combination for the # of processes(%d)", mpi.size);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}

		mpi.comm_c.size = mpi.size_2dr = twod_r;
		mpi.comm_r.size = mpi.size_2dc = twod_c;
		mpi.comm_c.rank = mpi.rank_2dr = mpi.rank % mpi.size_2dr;
		mpi.comm_r.rank = mpi.rank_2dc = mpi.rank / mpi.size_2dr;
	}

	mpi.isRowMajor = false;
	mpi.rank_2d = mpi.rank_2dr + mpi.rank_2dc * mpi.size_2dr;
	mpi.size_2d = mpi.size_2dr * mpi.size_2dc;
	MPI_Comm_split(MPI_COMM_WORLD, mpi.rank_2dc, mpi.rank_2dr, &mpi.comm_2dc);
	mpi.comm_c.comm = mpi.comm_2dc;
	MPI_Comm_split(MPI_COMM_WORLD, mpi.rank_2dr, mpi.rank_2dc, &mpi.comm_2dr);
	mpi.comm_r.comm = mpi.comm_2dr;
	MPI_Comm_split(MPI_COMM_WORLD, 0, mpi.rank_2d, &mpi.comm_2d);
}

void setup_globals(int argc, char** argv, int SCALE, int edgefactor)
{
	int reqeust_level = MPI_THREAD_FUNNELED;
	MPI_Init_thread(&argc, &argv, reqeust_level, &mpi.thread_level);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
	setup_2dcomm();

	// Initialize comm_[yz]
	mpi.comm_y = mpi.comm_2dc;
	mpi.comm_z = MPI_COMM_SELF;
	mpi.size_y = mpi.size_2dr;
	mpi.size_z = 1;
	mpi.rank_y = mpi.rank_2dr;
	mpi.rank_z = 0;

	// change default error handler
	MPI_File_set_errhandler(MPI_FILE_NULL, MPI_ERRORS_ARE_FATAL);

	UnweightedEdge::initialize();
	UnweightedPackedEdge::initialize();
}

namespace MpiCol {
template <typename T>
int allgatherv(T* sendbuf, T* recvbuf, int sendcount, MPI_Comm comm, int comm_size) {
	int recv_off[comm_size+1], recv_cnt[comm_size];
	MPI_Allgather(&sendcount, 1, MPI_INT, recv_cnt, 1, MPI_INT, comm);
	recv_off[0] = 0;
	for(int i = 0; i < comm_size; ++i) {
		recv_off[i+1] = recv_off[i] + recv_cnt[i];
	}
	MPI_Allgatherv(sendbuf, sendcount, MpiTypeOf<T>::type,
			recvbuf, recv_cnt, recv_off, MpiTypeOf<T>::type, comm);
	return recv_off[comm_size];
}

template <typename T>
void alltoall(T* sendbuf, T* recvbuf, int sendcount, MPI_Comm comm) {
	MPI_Alltoall(sendbuf, sendcount, MpiTypeOf<T>::type,
			recvbuf, sendcount, MpiTypeOf<T>::type, comm);
}

template <typename T>
T* alltoallv(T* sendbuf, int* sendcount,
		int* sendoffset, int* recvcount, int* recvoffset, MPI_Comm comm, int comm_size)
{
	sendoffset[0] = 0;
	for(int r = 0; r < comm_size; ++r) {
		sendoffset[r + 1] = sendoffset[r] + sendcount[r];
	}
	MPI_Alltoall(sendcount, 1, MPI_INT, recvcount, 1, MPI_INT, comm);
	// calculate offsets
	recvoffset[0] = 0;
	for(int r = 0; r < comm_size; ++r) {
		recvoffset[r + 1] = recvoffset[r] + recvcount[r];
	}
	T* recv_data = static_cast<T*>(xMPI_Alloc_mem(recvoffset[comm_size] * sizeof(T)));
	MPI_Alltoallv(sendbuf, sendcount, sendoffset, MpiTypeOf<T>::type,
			recv_data, recvcount, recvoffset, MpiTypeOf<T>::type, comm);
	return recv_data;
}
} // namespace MpiCol {

template <typename T>
class ParallelPartitioning
{
public:
	ParallelPartitioning(int num_partitions)
		: num_partitions_(num_partitions)
		, max_threads_(omp_get_max_threads())
		, thread_counts_(NULL)
		, thread_offsets_(NULL)
	{
		buffer_width_ = std::max<int>(CACHE_LINE/sizeof(T), num_partitions_);
		thread_counts_ = static_cast<T*>(cache_aligned_xmalloc(buffer_width_ * (max_threads_*2 + 1) * sizeof(T)));
		thread_offsets_ = thread_counts_ + buffer_width_*max_threads_;

		partition_size_ = static_cast<T*>(cache_aligned_xmalloc((num_partitions_*2 + 1) * sizeof(T)));
		partition_offsets_ = partition_size_ + num_partitions_;
	}
	~ParallelPartitioning()
	{
		free(thread_counts_);
		free(partition_size_);
	}
	T sum(T* base_offset = NULL) {
		const int width = buffer_width_;
		// compute sum of thread local count values
#pragma omp parallel for
		for(int r = 0; r < num_partitions_; ++r) {
			int sum = 0;
			for(int t = 0; t < max_threads_; ++t) {
				sum += thread_counts_[t*width + r];
			}
			partition_size_[r] = sum;
		}
		// compute offsets
		if(base_offset != NULL) {
#pragma omp parallel for
			for(int r = 0; r < num_partitions_; ++r) {
				partition_offsets_[r] = base_offset[r];
				base_offset[r] += partition_size_[r];
			}
#pragma omp parallel for
			for(int r = 0; r < num_partitions_; ++r) {
				thread_offsets_[0*width + r] = partition_offsets_[r];
				for(int t = 0; t < max_threads_; ++t) {
					thread_offsets_[(t+1)*width + r] = thread_offsets_[t*width + r] + thread_counts_[t*width + r];
				}
			}
			return T(0);
		}
		else {
			partition_offsets_[0] = 0;
			for(int r = 0; r < num_partitions_; ++r) {
				partition_offsets_[r + 1] = partition_offsets_[r] + partition_size_[r];
			}
			// assert (send_counts[size] == bufsize*2);
			// compute offset of each threads
	#pragma omp parallel for
			for(int r = 0; r < num_partitions_; ++r) {
				thread_offsets_[0*width + r] = partition_offsets_[r];
				for(int t = 0; t < max_threads_; ++t) {
					thread_offsets_[(t+1)*width + r] = thread_offsets_[t*width + r] + thread_counts_[t*width + r];
				}
				assert (thread_offsets_[max_threads_*width + r] == partition_offsets_[r + 1]);
			}
			return partition_offsets_[num_partitions_];
		}
	}
	T* get_counts() {
		T* counts = &thread_counts_[buffer_width_*omp_get_thread_num()];
		memset(counts, 0x00, buffer_width_*sizeof(T));
		return counts;
	}
	T* get_offsets() { return &thread_offsets_[buffer_width_*omp_get_thread_num()]; }

	const T* get_partition_offsets() const { return partition_offsets_; }
	const T* get_partition_size() const { return partition_size_; }

	bool check() const {
#ifndef	NDEBUG
		const int width = buffer_width_;
		// check offset of each threads
		for(int r = 0; r < num_partitions_; ++r) {
			assert (thread_offsets_[0*width + r] == partition_offsets_[r] + thread_counts_[0*width + r]);
			for(int t = 1; t < max_threads_; ++t) {
				assert (thread_offsets_[t*width + r] == thread_offsets_[(t-1)*width + r] + thread_counts_[t*width + r]);
			}
		}
#endif
		return true;
	}
private:
	int num_partitions_;
	int buffer_width_;
	int max_threads_;
	T* thread_counts_;
	T* thread_offsets_;
	T* partition_size_;
	T* partition_offsets_;
};

// Usage: get_counts -> sum -> get_offsets -> scatter -> gather
class ScatterContext
{
public:
	explicit ScatterContext(MPI_Comm comm)
		: comm_(comm)
		, max_threads_(omp_get_max_threads())
		, thread_counts_(NULL)
		, thread_offsets_(NULL)
		, send_counts_(NULL)
		, send_offsets_(NULL)
		, recv_counts_(NULL)
		, recv_offsets_(NULL)
	{
		MPI_Comm_size(comm_, &comm_size_);

		buffer_width_ = std::max<int>(CACHE_LINE/sizeof(int), comm_size_);
		thread_counts_ = static_cast<int*>(cache_aligned_xmalloc(buffer_width_ * (max_threads_*2 + 1) * sizeof(int)));
		thread_offsets_ = thread_counts_ + buffer_width_*max_threads_;

		send_counts_ = static_cast<int*>(cache_aligned_xmalloc((comm_size_*2 + 1) * 2 * sizeof(int)));
		send_offsets_ = send_counts_ + comm_size_;
		recv_counts_ = send_offsets_ + comm_size_ + 1;
		recv_offsets_ = recv_counts_ + comm_size_;
	}

	~ScatterContext()
	{
		::free(thread_counts_);
		::free(send_counts_);
	}

	int* get_counts() {
		int* counts = &thread_counts_[buffer_width_*omp_get_thread_num()];
		memset(counts, 0x00, buffer_width_*sizeof(int));
		return counts;
	}
	int* get_offsets() { return &thread_offsets_[buffer_width_*omp_get_thread_num()]; }

	void sum() {
		const int width = buffer_width_;
		// compute sum of thread local count values
#pragma omp parallel for if(comm_size_ > 1000)
		for(int r = 0; r < comm_size_; ++r) {
			int sum = 0;
			for(int t = 0; t < max_threads_; ++t) {
				sum += thread_counts_[t*width + r];
			}
			send_counts_[r] = sum;
		}
		// compute offsets
		send_offsets_[0] = 0;
		for(int r = 0; r < comm_size_; ++r) {
			send_offsets_[r + 1] = send_offsets_[r] + send_counts_[r];
		}
		// assert (send_counts[size] == bufsize*2);
		// compute offset of each threads
#pragma omp parallel for if(comm_size_ > 1000)
		for(int r = 0; r < comm_size_; ++r) {
			thread_offsets_[0*width + r] = send_offsets_[r];
			for(int t = 0; t < max_threads_; ++t) {
				thread_offsets_[(t+1)*width + r] = thread_offsets_[t*width + r] + thread_counts_[t*width + r];
			}
			assert (thread_offsets_[max_threads_*width + r] == send_offsets_[r + 1]);
		}
	}

	int get_send_count() { return send_offsets_[comm_size_]; }
	int get_recv_count() { return recv_offsets_[comm_size_]; }
	int* get_recv_offsets() { return recv_offsets_; }

	template <typename T>
	T* scatter(T* send_data) {
#ifndef	NDEBUG
		const int width = buffer_width_;
		// check offset of each threads
		for(int r = 0; r < comm_size_; ++r) {
			assert (thread_offsets_[0*width + r] == send_offsets_[r] + thread_counts_[0*width + r]);
			for(int t = 1; t < max_threads_; ++t) {
				assert (thread_offsets_[t*width + r] == thread_offsets_[(t-1)*width + r] + thread_counts_[t*width + r]);
			}
		}
#endif
		return MpiCol::alltoallv(send_data, send_counts_, send_offsets_,
				recv_counts_, recv_offsets_, comm_, comm_size_);
	}

	template <typename T>
	T* gather(T* send_data) {
		T* recv_data = static_cast<T*>(xMPI_Alloc_mem(send_offsets_[comm_size_] * sizeof(T)));
		MPI_Alltoallv(send_data, recv_counts_, recv_offsets_, MpiTypeOf<T>::type,
				recv_data, send_counts_, send_offsets_, MpiTypeOf<T>::type, comm_);
		return recv_data;
	}

	template <typename T>
	void free(T* buffer) {
		MPI_Free_mem(buffer);
	}

	void alltoallv(void* sendbuf, void* recvbuf, MPI_Datatype type, int recvbufsize)
	{
		recv_offsets_[0] = 0;
		for(int r = 0; r < comm_size_; ++r) {
			recv_offsets_[r + 1] = recv_offsets_[r] + recv_counts_[r];
		}
		MPI_Alltoall(send_counts_, 1, MPI_INT, recv_counts_, 1, MPI_INT, comm_);
		// calculate offsets
		recv_offsets_[0] = 0;
		for(int r = 0; r < comm_size_; ++r) {
			recv_offsets_[r + 1] = recv_offsets_[r] + recv_counts_[r];
		}
		if(recv_counts_[comm_size_] > recvbufsize) {
			fprintf(IMD_OUT, "Error: recv_counts_[comm_size_] > recvbufsize");
			throw "Error: buffer size not enough";
		}
		MPI_Alltoallv(sendbuf, send_counts_, send_offsets_, type,
				recvbuf, recv_counts_, recv_offsets_, type, comm_);
	}

private:
	MPI_Comm comm_;
	int comm_size_;
	int buffer_width_;
	int max_threads_;
	int* thread_counts_;
	int* thread_offsets_;
	int* restrict send_counts_;
	int* restrict send_offsets_;
	int* restrict recv_counts_;
	int* restrict recv_offsets_;
};

namespace MpiCol {
template <typename Mapping>
void scatter(const Mapping mapping, int data_count, MPI_Comm comm)
{
	ScatterContext scatter(comm);
	typename Mapping::send_type* restrict partitioned_data = static_cast<typename Mapping::send_type*>(
						cache_aligned_xmalloc(data_count*sizeof(typename Mapping::send_type)));
#pragma omp parallel
	{
		int* restrict counts = scatter.get_counts();

#pragma omp for schedule(static)
		for (int i = 0; i < data_count; ++i) {
			(counts[mapping.target(i)])++;
		} // #pragma omp for schedule(static)
	}

	scatter.sum();

#pragma omp parallel
	{
		int* restrict offsets = scatter.get_offsets();

#pragma omp for schedule(static)
		for (int i = 0; i < data_count; ++i) {
			partitioned_data[(offsets[mapping.target(i)])++] = mapping.get(i);
		} // #pragma omp for schedule(static)
	} // #pragma omp parallel

	typename Mapping::send_type* recv_data = scatter.scatter(partitioned_data);
	int recv_count = scatter.get_recv_count();
	::free(partitioned_data); partitioned_data = NULL;

	int i;
#pragma omp parallel for lastprivate(i) schedule(static)
	for(i = 0; i < (recv_count&(~3)); i += 4) {
		mapping.set(i+0, recv_data[i+0]);
		mapping.set(i+1, recv_data[i+1]);
		mapping.set(i+2, recv_data[i+2]);
		mapping.set(i+3, recv_data[i+3]);
	} // #pragma omp parallel for
	for( ; i < recv_count; ++i) {
		mapping.set(i, recv_data[i]);
	}

	scatter.free(recv_data);
}

template <typename Mapping>
void gather(const Mapping mapping, int data_count, MPI_Comm comm)
{
	ScatterContext scatter(comm);

	int* restrict local_indices = static_cast<int*>(
			cache_aligned_xmalloc(data_count*sizeof(int)));
	typename Mapping::send_type* restrict partitioned_data = static_cast<typename Mapping::send_type*>(
			cache_aligned_xmalloc(data_count*sizeof(typename Mapping::send_type)));

#pragma omp parallel
	{
		int* restrict counts = scatter.get_counts();

#pragma omp for schedule(static)
		for (int i = 0; i < data_count; ++i) {
			(counts[mapping.target(i)])++;
		} // #pragma omp for schedule(static)
	}

	scatter.sum();

#pragma omp parallel
	{
		int* restrict offsets = scatter.get_offsets();

#pragma omp for schedule(static)
		for (int i = 0; i < data_count; ++i) {
			int pos = (offsets[mapping.target(i)])++;
			assert (pos < data_count);
			local_indices[i] = pos;
			partitioned_data[pos] = mapping.get(i);
			//// user defined ////
		} // #pragma omp for schedule(static)
	} // #pragma omp parallel

	// send and receive requests
	typename Mapping::send_type* restrict reply_verts = scatter.scatter(partitioned_data);
	int recv_count = scatter.get_recv_count();
	::free(partitioned_data);

	// make reply data
	typename Mapping::recv_type* restrict reply_data = static_cast<typename Mapping::recv_type*>(
			cache_aligned_xmalloc(recv_count*sizeof(typename Mapping::recv_type)));
#pragma omp parallel for
	for (int i = 0; i < recv_count; ++i) {
		reply_data[i] = mapping.map(reply_verts[i]);
	}
	scatter.free(reply_verts);

	// send and receive reply
	typename Mapping::recv_type* restrict recv_data = scatter.gather(reply_data);
	::free(reply_data);

	// apply received data to edges
#pragma omp parallel for
	for (int i = 0; i < data_count; ++i) {
		mapping.set(i, recv_data[local_indices[i]]);
	}

	scatter.free(recv_data);
	::free(local_indices);
}

} // namespace MpiCollective { //

double to_mega(int64_t v) { return v / (1024.0*1024.0); }
template <typename T> struct TypeName { };
template <> struct TypeName<int8_t> { static const char* value; };
const char* TypeName<int8_t>::value = "int8_t";
template <> struct TypeName<uint8_t> { static const char* value; };
const char* TypeName<uint8_t>::value = "uint8_t";
template <> struct TypeName<int16_t> { static const char* value; };
const char* TypeName<int16_t>::value = "int16_t";
template <> struct TypeName<uint16_t> { static const char* value; };
const char* TypeName<uint16_t>::value = "uint16_t";
template <> struct TypeName<int32_t> { static const char* value; };
const char* TypeName<int32_t>::value = "int32_t";
template <> struct TypeName<uint32_t> { static const char* value; };
const char* TypeName<uint32_t>::value = "uint32_t";
template <> struct TypeName<int64_t> { static const char* value; };
const char* TypeName<int64_t>::value = "int64_t";
template <> struct TypeName<uint64_t> { static const char* value; };
const char* TypeName<uint64_t>::value = "uint64_t";

namespace memory {

template <typename T>
class Pool {
public:
	Pool()
	{
	}
	virtual ~Pool() {
		clear_();
	}

	virtual T* get() {
		if(free_list_.empty()) {
			return allocate_new();
		}
		T* buffer = free_list_.back();
		free_list_.pop_back();
		return buffer;
	}

	virtual void free(T* buffer) {
		free_list_.push_back(buffer);
	}

	virtual void clear() {
		clear_();
	}

	bool empty() const {
		return free_list_.size() == 0;
	}

	size_t size() const {
		return free_list_.size();
	}

protected:
	std::vector<T*> free_list_;

	virtual T* allocate_new() {
		return new (malloc(sizeof(T))) T();
	}

private:
	void clear_() {
		for(int i = 0; i < (int)free_list_.size(); ++i) {
			free_list_[i]->~T();
			::free(free_list_[i]);
		}
		free_list_.clear();
	}
};

//! Only get() and free() are thread-safe. The other functions are NOT thread-safe.
template <typename T>
class ConcurrentPool : public Pool<T> {
	typedef Pool<T> super_;
public:
	ConcurrentPool()
		: Pool<T>()
	{
		pthread_mutex_init(&thread_sync_, NULL);
	}
	virtual ~ConcurrentPool()
	{
		pthread_mutex_lock(&thread_sync_);
	}

	virtual T* get() {
		pthread_mutex_lock(&thread_sync_);
		if(this->free_list_.empty()) {
			pthread_mutex_unlock(&thread_sync_);
			T* new_buffer = this->allocate_new();
			return new_buffer;
		}
		T* buffer = this->free_list_.back();
		this->free_list_.pop_back();
		pthread_mutex_unlock(&thread_sync_);
		return buffer;
	}

	virtual void free(T* buffer) {
		pthread_mutex_lock(&thread_sync_);
		this->free_list_.push_back(buffer);
		pthread_mutex_unlock(&thread_sync_);
	}

	virtual void clear() {
		pthread_mutex_lock(&thread_sync_);
		super_::clear();
		pthread_mutex_unlock(&thread_sync_);
	}

	/*
	bool empty() const { return super_::empty(); }
	size_t size() const { return super_::size(); }
	void clear() { super_::clear(); }
	*/
protected:
	pthread_mutex_t thread_sync_;
};


template <typename T>
class Store {
public:
	Store() {
	}
	void init(Pool<T>* pool) {
		pool_ = pool;
		filled_length_ = 0;
		buffer_length_ = 0;
		resize_buffer(16);
	}
	~Store() {
		for(int i = 0; i < filled_length_; ++i){
			pool_->free(buffer_[i]);
		}
		filled_length_ = 0;
		buffer_length_ = 0;
		::free(buffer_); buffer_ = NULL;
	}

	void submit(T* value) {
		const int offset = filled_length_++;

		if(buffer_length_ == filled_length_)
			expand();

		buffer_[offset] = value;
	}

	void clear() {
		for(int i = 0; i < filled_length_; ++i){
			buffer_[i]->clear();
			assert (buffer_[i]->size() == 0);
			pool_->free(buffer_[i]);
		}
		filled_length_ = 0;
	}

	T* front() {
		if(filled_length_ == 0) {
			push();
		}
		return buffer_[filled_length_ - 1];
	}

	void push() {
		submit(pool_->get());
	}

	int64_t size() const { return filled_length_; }
	T* get(int index) const { return buffer_[index]; }
private:

	void resize_buffer(int allocation_size)
	{
		T** new_buffer = (T**)malloc(allocation_size*sizeof(buffer_[0]));
		if(buffer_length_ != 0) {
			memcpy(new_buffer, buffer_, filled_length_*sizeof(buffer_[0]));
			::free(buffer_);
		}
		buffer_ = new_buffer;
		buffer_length_ = allocation_size;
	}

	void expand()
	{
		if(filled_length_ == buffer_length_)
			resize_buffer(std::max<int64_t>(buffer_length_*2, 16));
	}

	int64_t filled_length_;
	int64_t buffer_length_;
	T** buffer_;
	Pool<T>* pool_;
};

template <typename T>
class ConcurrentStack
{
public:
	ConcurrentStack()
	{
		pthread_mutex_init(&thread_sync_, NULL);
	}

	~ConcurrentStack()
	{
		pthread_mutex_destroy(&thread_sync_);
	}

	void push(const T& d)
	{
		pthread_mutex_lock(&thread_sync_);
		stack_.push_back(d);
		pthread_mutex_unlock(&thread_sync_);
	}

	bool pop(T* ret)
	{
		pthread_mutex_lock(&thread_sync_);
		if(stack_.size() == 0) {
			pthread_mutex_unlock(&thread_sync_);
			return false;
		}
		*ret = stack_.back(); stack_.pop_back();
		pthread_mutex_unlock(&thread_sync_);
		return true;
	}

	std::vector<T> stack_;
	pthread_mutex_t thread_sync_;
};

struct SpinBarrier {
	volatile int step, cnt;
	int max;
	explicit SpinBarrier(int num_threads) {
		step = cnt = 0;
		max = num_threads;
	}
  void barrier() {
		int cur_step = step;
		int wait_cnt = __sync_add_and_fetch(&cnt, 1);
		assert (wait_cnt <= max);
		if(wait_cnt == max) {
			cnt = 0;
			__sync_add_and_fetch(&step, 1);
			return ;
		}
		while(step == cur_step) ;
	}
};

void copy_mt(void* dst, void* src, size_t size) {
#pragma omp parallel
	{
		int num_threads = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int64_t i_start, i_end;
		get_partition<int64_t>(size, num_threads, tid, i_start, i_end);
		memcpy((int8_t*)dst + i_start, (int8_t*)src + i_start, i_end - i_start);
	}
}
} // namespace memory

#endif /* UTILS_IMPL_HPP_ */

