/*
 * utils.hpp
 *
 *  Created on: Dec 9, 2011
 *      Author: koji
 */

#ifndef UTILS_IMPL_HPP_
#define UTILS_IMPL_HPP_

#include <stdint.h>

// for affinity setting //
#include <unistd.h>

#ifndef NNUMA
#include <sched.h>
#include <numa.h>
#endif

#include <omp.h>

#include <sys/types.h>
#include <sys/time.h>
#include <sys/shm.h>

#include <algorithm>
#include <vector>
#include <deque>

#include "mpi_workarounds.h"
#include "utils_core.h"
#include "primitives.hpp"
#if CUDA_ENABLED
#include "gpu_host.hpp"
#endif

struct MPI_GLOBALS {
	int rank;
	int size_;

	// 2D
	int rank_2d;
	int rank_2dr;
	int rank_2dc;
	int rank_y;
	int rank_z;
	int size_2d;
	int size_2dc;
	int size_2dr;
	int size_y;
	int size_z;
	MPI_Comm comm_2d;
	MPI_Comm comm_2dr;
	MPI_Comm comm_2dc;
	MPI_Comm comm_y;
	MPI_Comm comm_z;
	bool isPadding2D;
	bool isRowMajor;

	// utility method
	bool isMaster() const { return rank == 0; }
	bool isRmaster() const { return rank == size_-1; }
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

//-------------------------------------------------------------//
// Memory Allocation
//-------------------------------------------------------------//

void* xMPI_Alloc_mem(size_t nbytes) {
  void* p;
  MPI_Alloc_mem(nbytes, MPI_INFO_NULL, &p);
  if (nbytes != 0 && !p) {
    fprintf(IMD_OUT, "MPI_Alloc_mem failed for size%zu (%"PRId64") byte(s)\n", nbytes, (int64_t)nbytes);
    throw "OutOfMemoryExpception";
  }
  return p;
}

void* cache_aligned_xcalloc(const size_t size)
{
	void* p;
	if(posix_memalign(&p, CACHE_LINE, size)){
		fprintf(IMD_OUT, "Out of memory trying to allocate %zu (%"PRId64") byte(s)\n", size, (int64_t)size);
		throw "OutOfMemoryExpception";
	}
	memset(p, 0, size);
	return p;
}
void* cache_aligned_xmalloc(const size_t size)
{
	void* p;
	if(posix_memalign(&p, CACHE_LINE, size)){
		fprintf(IMD_OUT, "Out of memory trying to allocate %zu (%"PRId64") byte(s)\n", size, (int64_t)size);
		throw "OutOfMemoryExpception";
	}
	return p;
}

void* page_aligned_xcalloc(const size_t size)
{
	void* p;
	if(posix_memalign(&p, PAGE_SIZE, size)){
		fprintf(IMD_OUT, "Out of memory trying to allocate %zu (%"PRId64") byte(s)\n", size, (int64_t)size);
		throw "OutOfMemoryExpception";
	}
	memset(p, 0, size);
	return p;
}
void* page_aligned_xmalloc(const size_t size)
{
	void* p;
	if(posix_memalign(&p, PAGE_SIZE, size)){
		fprintf(IMD_OUT, "Out of memory trying to allocate %zu (%"PRId64") byte(s)\n", size, (int64_t)size);
		throw "OutOfMemoryExpception";
	}
	return p;
}

void* shared_malloc(MPI_Comm comm, size_t nbytes) {
	int rank; MPI_Comm_rank(comm, &rank);
	key_t shm_key;
	int shmid;
	void* addr = NULL;

	if(rank == 0) {
		timeval tv; gettimeofday(&tv, NULL);
		shm_key = tv.tv_usec;
		for(int i = 0; i < 1000; ++i) {
			shmid = shmget(++shm_key, nbytes,
					IPC_CREAT | IPC_EXCL | 0600);
			if(shmid != -1) break;
#ifndef NDEBUG
			perror("shmget try");
#endif
		}
		if(shmid == -1) {
			perror("shmget");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		addr = shmat(shmid, NULL, 0);
		if(addr == (void*)-1) {
			perror("Shared memory attach failure");
			addr = NULL;
		}
	}

	MPI_Bcast(&shm_key, 1, MpiTypeOf<key_t>::type, 0, comm);

	if(rank != 0) {
		shmid = shmget(shm_key, 0, 0);
		if(shmid == -1) {
			perror("shmget");
		}
		else {
		addr = shmat(shmid, NULL, 0);
			if(addr == (void*)-1) {
				perror("Shared memory attach failure");
				addr = NULL;
			}
		}
	}

	MPI_Barrier(comm);

	if(rank == 0) {
		// release the memory when the last process is detached.
		if(shmctl(shmid, IPC_RMID, NULL) == -1) {
			perror("shmctl(shmid, IPC_RMID, NULL)");
		}
	}
	return addr;
}

void shared_free(void* shm) {
	if(shmdt(shm) == -1) {
		perror("shmdt(shm)");
	}
}

//-------------------------------------------------------------//
// CPU Affinity Setting
//-------------------------------------------------------------//

int g_GpuIndex = -1;

#ifndef NNUMA
typedef struct cpuid_register_t {
    unsigned long eax;
    unsigned long ebx;
    unsigned long ecx;
    unsigned long edx;
} cpuid_register_t;

void cpuid(unsigned int eax, cpuid_register_t *r)
{
    __asm__ volatile (
        "cpuid"
        :"=a"(r->eax), "=b"(r->ebx), "=c"(r->ecx), "=d"(r->edx)
        :"a"(eax)
    );
    return;
}

#define CPU_TO_PROC_MAP 2

const int cpu_to_proc_map2[] = { 0,0,0,0, 0,0,0,0, 0,0,0,0, 1,1,1,1, 1,1,1,1, 1,1,1,1 };

#if CPU_TO_PROC_MAP == 1
const int cpu_to_proc_map[] = { 9,9,9,9, 9,9,9,9, 9,9,9,9, 9,9,9,9, 9,9,9,9, 9,9,9,9 };
#elif CPU_TO_PROC_MAP == 2
const int * const cpu_to_proc_map = cpu_to_proc_map2;
#elif CPU_TO_PROC_MAP == 3
const int cpu_to_proc_map[] = { 0,0,0,0, 0,0,0,0, 2,2,2,2, 2,2,2,2, 1,1,1,1, 1,1,1,1 };
#elif CPU_TO_PROC_MAP == 4
const int cpu_to_proc_map[] = { 0,0,0,0, 0,2,0,2, 0,2,0,2, 2,1,2,1, 2,1,2,1, 1,1,1,1 };
#endif

void testSharedMemory() {
	int* mem = shared_malloc(mpi.comm_z, sizeof(int));
	int ref_val = 0;
	if(mpi.rank_z == 0) {
		*mem = ref_val = mpi.rank;
	}
	MPI_Bcast(&ref_val, 1, MpiTypeOf<int>::type, 0, mpi.comm_z);
	int result = (*mem == ref_val), global_result;
	shared_free(mem);
	MPI_Allreduce(&result, &global_result, 1, MpiTypeOf<int>::type, MPI_LOR, mpi.comm_2d);
	if(global_result == false) {
		if(mpi.isMaster()) fprintf(IMD_OUT, "Shared memory test failed!! Please, check MPI_NUM_NODE.\n");
		MPI_Abort(mpi.comm_2d, 1);
	}
}

void setAffinity()
{
	int NUM_PROCS = sysconf(_SC_NPROCESSORS_CONF);
	cpu_set_t set;
	int i;

	// Initialize comm_[yz]
	mpi.comm_y = mpi.comm_2dc;
	mpi.comm_z = MPI_COMM_SELF;
	mpi.size_y = mpi.size_2dr;
	mpi.size_z = 1;
	mpi.rank_y = mpi.rank_2dr;
	mpi.rank_z = 0;

	const char* num_node_str = getenv("MPI_NUM_NODE");
	if(num_node_str == NULL) {
		if(mpi.rank == mpi.size_ - 1) {
			fprintf(IMD_OUT, "Error: failed to get # of node. Please set MPI_NUM_NODE=<# of node>\n");
		}
		return ;
	}
	const char* dist_round_robin = getenv("MPI_ROUND_ROBIN");
	int num_node = atoi(num_node_str);

	int32_t core_list[NUM_PROCS];
	for(i = 0; i < NUM_PROCS; i++) {
		CPU_ZERO(&set);
		CPU_SET(i, &set);
		sched_setaffinity(0, sizeof(set), &set);
		sleep(0);
		cpuid_register_t reg;
		cpuid(1, &reg);
		int apicid = (reg.ebx >> 24) & 0xFF;
	//	fprintf(IMD_OUT, "%d-th -> apicid=%d\n", i, apicid);
		core_list[i] = (apicid << 16) | i;
	}

	std::sort(core_list ,core_list + NUM_PROCS);
#if 0
	fprintf(IMD_OUT, "sort\n");
	for(int i = 0; i < NUM_PROCS; i++) {
		int core_id = core_list[i] & 0xFFFF;
		int apicid = core_list[i] >> 16;
		fprintf(IMD_OUT, "%d-th -> apicid=%d\n", core_id, apicid);
	}
#endif
	int max_procs_per_node = (mpi.size_ + num_node - 1) / num_node;
	int proc_num = (dist_round_robin ? (mpi.rank / num_node) : (mpi.rank % max_procs_per_node));
	g_GpuIndex = proc_num;
#if CPU_TO_PROC_MAP != 2
	int node_num = mpi.rank % num_node;
	int split = ((mpi.size_ - 1) % num_node) + 1;
#endif

	if(mpi.isRmaster()) {
		fprintf(IMD_OUT, "process distribution : %s\n", dist_round_robin ? "round robin" : "partition");
	}
//#if SET_AFFINITY
	if(max_procs_per_node == 3) {
		if(numa_available() < 0) {
			fprintf(IMD_OUT, "No NUMA support available on this system.\n");
		}
		else {
			int NUM_SOCKET = numa_max_node() + 1;
			if(proc_num < NUM_SOCKET) {
				//numa_run_on_node(proc_num);
				numa_set_preferred(proc_num);
			}
		}

		CPU_ZERO(&set);
		int enabled = 0;
#if CPU_TO_PROC_MAP != 2
		const int *cpu_map = (dist_round_robin && node_num >= split) ? cpu_to_proc_map2 : cpu_to_proc_map;
#else
		const int *cpu_map = cpu_to_proc_map;
#endif
		for(i = 0; i < NUM_PROCS; i++) {
			if(cpu_map[i] == proc_num) {
				int core_id = core_list[i] & 0xFFFF;
				CPU_SET(core_id, &set);
				enabled = 1;
			}
		}
		if(enabled == 0) {
			for(i = 0; i < NUM_PROCS; i++) {
				CPU_SET(i, &set);
			}
		}
		sched_setaffinity(0, sizeof(set), &set);

		if(mpi.isRmaster()) { /* print from max rank node for easy debugging */
		  fprintf(IMD_OUT, "affinity for executing 3 processed per node is enabled.\n");
		}
	}
	else if(max_procs_per_node > 1) {
		if(numa_available() < 0) {
			fprintf(IMD_OUT, "No NUMA support available on this system.\n");
			return ;
		}
		int NUM_SOCKET = numa_max_node() + 1;

		// create comm_z
		mpi.size_z = mpi.size_ / num_node;
		if(dist_round_robin) {
			mpi.rank_z = mpi.rank / num_node;
			MPI_Comm_split(mpi.comm_2d, mpi.rank % num_node, mpi.rank_z, &mpi.comm_z);
		}
		else {
			mpi.rank_z = mpi.rank % max_procs_per_node;
			MPI_Comm_split(mpi.comm_2d, mpi.rank / max_procs_per_node, mpi.rank_z, &mpi.comm_z);
		}
		numa_run_on_node(mpi.rank_z % NUM_SOCKET);
		numa_set_preferred(mpi.rank_z % NUM_SOCKET);

		// test shared memory
		testSharedMemory();

		// create comm_y
		if(dist_round_robin == false && mpi.isRowMajor == false) {
			mpi.rank_y = mpi.rank_2dc / mpi.size_z;
			mpi.size_y = mpi.size_2dr / mpi.size_z;
			MPI_Comm_split(mpi.comm_2dc, mpi.rank_z, mpi.rank_2dc / mpi.size_z, &mpi.comm_y);
		}

		if(mpi.rank == mpi.size_-1) { /* print from max rank node for easy debugging */
		  fprintf(IMD_OUT, "NUMA node affinity is enabled.\n");
		}
	}
	else
//#endif
	{
		//
		if(mpi.isRmaster()) { /* print from max rank node for easy debugging */
		  fprintf(IMD_OUT, "affinity is disabled.\n");
		}
		CPU_ZERO(&set);
		for(i = 0; i < NUM_PROCS; i++) {
			CPU_SET(i, &set);
		}
		sched_setaffinity(0, sizeof(set), &set);
	}
	if(mpi.isMaster()) {
		  fprintf(IMD_OUT, "Y dimension is %s\n", mpi.isYdimAvailable() ? "Enabled" : "Disabled");
	}
}
#endif

//-------------------------------------------------------------//
// ?
//-------------------------------------------------------------//

static void setup_2dcomm(bool row_major)
{
	const int log_size = get_msb_index(mpi.size_);

	const char* twod_r_str = getenv("TWOD_R");
	int log_size_r = log_size / 2;
	if(twod_r_str){
		int twod_r = atoi((char*)twod_r_str);
		if(twod_r == 0 || /* Check for power of 2 */ (twod_r & (twod_r - 1)) != 0) {
			fprintf(IMD_OUT, "Number of Rows %d is not a power of two.\n", twod_r);
		}
		else {
			log_size_r = get_msb_index(twod_r);
		}
	}

	int log_size_c = log_size - log_size_r;

	mpi.size_2dr = (1 << log_size_r);
	mpi.size_2dc = (1 << log_size_c);

	if(row_major) {
		// row major
		mpi.rank_2dr = mpi.rank / mpi.size_2dc;
		mpi.rank_2dc = mpi.rank % mpi.size_2dc;
	}
	else {
		// column major
		mpi.rank_2dr = mpi.rank / mpi.size_2dr;
		mpi.rank_2dc = mpi.rank % mpi.size_2dr;
	}

	mpi.rank_2d = mpi.rank_2dr + mpi.rank_2dc * mpi.size_2dr;
	mpi.size_2d = mpi.size_2dr * mpi.size_2dc;
	MPI_Comm_split(MPI_COMM_WORLD, 0, mpi.rank_2d, &mpi.comm_2d);
	MPI_Comm_split(MPI_COMM_WORLD, mpi.rank_2dc, mpi.rank_2dr, &mpi.comm_2dc);
	MPI_Comm_split(MPI_COMM_WORLD, mpi.rank_2dr, mpi.rank_2dc, &mpi.comm_2dr);
	mpi.isRowMajor = row_major;
}

// assume rank = XYZ
static void setup_2dcomm_on_3d()
{
	const int log_size = get_msb_index(mpi.size_);

	const char* treed_map_str = getenv("TREED_MAP");
	if(treed_map_str) {
		int X, Y, Z1, Z2, A, B;
		sscanf(treed_map_str, "%dx%dx%dx%d", &X, &Y, &Z1, &Z2);
		A = X * Z1;
		B = Y * Z2;
		mpi.size_2dr = 1 << get_msb_index(A);
		mpi.size_2dc = 1 << get_msb_index(B);

		if(mpi.isMaster()) fprintf(IMD_OUT, "Dimension: (%dx%dx%dx%d) -> (%dx%d) -> (%dx%d)\n", X, Y, Z1, Z2, A, B, mpi.size_2dr, mpi.size_2dc);
		if(mpi.size_ < A*B) {
			if(mpi.isMaster()) fprintf(IMD_OUT, "Error: There are not enough processes.\n");
		}

		int x, y, z1, z2;
		x = mpi.rank % X;
		y = (mpi.rank / X) % Y;
		z1 = (mpi.rank / (X*Y)) % Z1;
		z2 = mpi.rank / (X*Y*Z1);
		mpi.rank_2dr = z1 * X + x;
		mpi.rank_2dc = z2 * Y + y;

		mpi.rank_2d = mpi.rank_2dr + mpi.rank_2dc * mpi.size_2dr;
		mpi.size_2d = mpi.size_2dr * mpi.size_2dc;
		mpi.isPadding2D = (mpi.rank_2dr >= mpi.size_2dr || mpi.rank_2dc >= mpi.size_2dc) ? true : false;
		MPI_Comm_split(MPI_COMM_WORLD, mpi.isPadding2D ? 1 : 0, mpi.rank_2d, &mpi.comm_2d);
		if(mpi.isPadding2D == false) {
			MPI_Comm_split(mpi.comm_2d, mpi.rank_2dc, mpi.rank_2dr, &mpi.comm_2dc);
			MPI_Comm_split(mpi.comm_2d, mpi.rank_2dr, mpi.rank_2dc, &mpi.comm_2dr);
		}
	}
	else if((1 << log_size) != mpi.size_) {
		if(mpi.isMaster()) fprintf(IMD_OUT, "The program needs dimension information when mpi processes is not a power of two.\n");
	}

}

void cleanup_2dcomm()
{
	MPI_Comm_free(&mpi.comm_2d);
	if(mpi.isPadding2D == false) {
		MPI_Comm_free(&mpi.comm_2dr);
		MPI_Comm_free(&mpi.comm_2dc);
	}
}

void setup_globals(int argc, char** argv, int SCALE, int edgefactor)
{
	{
		int prov;
		MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &prov);
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
		MPI_Comm_size(MPI_COMM_WORLD, &mpi.size_);
	}

	if(mpi.isMaster()) {
		fprintf(IMD_OUT, "Graph500 Benchmark: SCALE: %d, edgefactor: %d %s\n", SCALE, edgefactor,
#ifdef NDEBUG
				""
#else
				"(Debug Mode)"
#endif
		);
		fprintf(IMD_OUT, "Running Binary: %s\n", argv[0]);
		fprintf(IMD_OUT, "Pre running time will be %d seconds\n", PRE_EXEC_TIME);
	}

	if(getenv("TREED_MAP")) {
		setup_2dcomm_on_3d();
	}
	else {
		setup_2dcomm(false);
	}

	// enables nested
	omp_set_nested(1);

	// change default error handler
	MPI_File_set_errhandler(MPI_FILE_NULL, MPI_ERRORS_ARE_FATAL);

#ifdef _OPENMP
	if(mpi.isRmaster()){
#if _OPENMP >= 200805
	  omp_sched_t kind;
	  int modifier;
	  omp_get_schedule(&kind, &modifier);
	  const char* kind_str = "unknown";
	  switch(kind) {
		case omp_sched_static:
		  kind_str = "omp_sched_static";
		  break;
		case omp_sched_dynamic:
		  kind_str = "omp_sched_dynamic";
		  break;
		case omp_sched_guided:
		  kind_str = "omp_sched_guided";
		  break;
		case omp_sched_auto:
		  kind_str = "omp_sched_auto";
		  break;
	  }
	  fprintf(IMD_OUT, "OpenMP default scheduling : %s, %d\n", kind_str, modifier);
#else
	  fprintf(IMD_OUT, "OpenMP version : %d\n", _OPENMP);
#endif
	}
#endif

	UnweightedEdge::initialize();
	UnweightedPackedEdge::initialize();
	WeightedEdge::initialize();

	// check page size
	if(mpi.isMaster()) {
		long page_size = sysconf(_SC_PAGESIZE);
		if(page_size != PAGE_SIZE) {
			fprintf(IMD_OUT, "System Page Size: %ld\n", page_size);
			fprintf(IMD_OUT, "Error: PAGE_SIZE(%d) is not correct.\n", PAGE_SIZE);
		}
	}
#ifndef NNUMA
	// set affinity
	if(getenv("NO_AFFINITY") == NULL) {
		setAffinity();
	}
#endif

#if CUDA_ENABLED
	CudaStreamManager::initialize_cuda(g_GpuIndex);

	MPI_INFO_ON_GPU mpig;
	mpig.rank = mpi.rank;
	mpig.size = mpi.size_;
	mpig.rank_2d = mpi.rank_2d;
	mpig.rank_2dr = mpi.rank_2dr;
	mpig.rank_2dc = mpi.rank_2dc;
	CudaStreamManager::begin_cuda();
	CUDA_CHECK(cudaMemcpyToSymbol("mpig", &mpig, sizeof(mpig), 0, cudaMemcpyHostToDevice));
	CudaStreamManager::end_cuda();
#endif
}

void cleanup_globals()
{
	cleanup_2dcomm();

	UnweightedEdge::uninitialize();
	UnweightedPackedEdge::uninitialize();
	WeightedEdge::uninitialize();

#if CUDA_ENABLED
	CudaStreamManager::finalize_cuda();
#endif

	MPI_Finalize();
}

//-------------------------------------------------------------//
// Multithread Partitioning and Scatter
//-------------------------------------------------------------//

// Usage: get_counts -> sum -> get_offsets
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
	T sum() {
		const int width = buffer_width_;
		// compute sum of thread local count values
		for(int r = 0; r < num_partitions_; ++r) {
			int sum = 0;
			for(int t = 0; t < max_threads_; ++t) {
				sum += thread_counts_[t*width + r];
			}
			partition_size_[r] = sum;
		}
		// compute offsets
		partition_offsets_[0] = 0;
		for(int r = 0; r < num_partitions_; ++r) {
			partition_offsets_[r + 1] = partition_offsets_[r] + partition_size_[r];
		}
		// assert (send_counts[size] == bufsize*2);
		// compute offset of each threads
		for(int r = 0; r < num_partitions_; ++r) {
			thread_offsets_[0*width + r] = partition_offsets_[r];
			for(int t = 0; t < max_threads_; ++t) {
				thread_offsets_[(t+1)*width + r] = thread_offsets_[t*width + r] + thread_counts_[t*width + r];
			}
			assert (thread_offsets_[max_threads_*width + r] == partition_offsets_[r + 1]);
		}
		return partition_offsets_[num_partitions_];
	}
	T* get_counts() {
		T* counts = &thread_counts_[buffer_width_*omp_get_thread_num()];
		memset(counts, 0x00, buffer_width_*sizeof(T));
		return counts;
	}
	T* get_offsets() { return &thread_offsets_[buffer_width_*omp_get_thread_num()]; }

	const T* get_partition_offsets() const { return partition_offsets_; }

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
#if NETWORK_PROBLEM_AYALISYS
		if(mpi.isMaster()) fprintf(IMD_OUT, "MPI_Alltoall(MPI_INT, comm_size=%d)...\n", comm_size_);
#endif
		MPI_Alltoall(send_counts_, 1, MPI_INT, recv_counts_, 1, MPI_INT, comm_);
#if NETWORK_PROBLEM_AYALISYS
		if(mpi.isMaster()) fprintf(IMD_OUT, "OK\n");
#endif
		// calculate offsets
		recv_offsets_[0] = 0;
		for(int r = 0; r < comm_size_; ++r) {
			recv_offsets_[r + 1] = recv_offsets_[r] + recv_counts_[r];
		}
		T* recv_data = static_cast<T*>(xMPI_Alloc_mem(recv_offsets_[comm_size_] * sizeof(T)));
#if NETWORK_PROBLEM_AYALISYS
		if(mpi.isMaster()) fprintf(IMD_OUT, "MPI_Alltoallv(send_offsets_[%d]=%d\n", comm_size_, send_offsets_[comm_size_]);
#endif
		MPI_Alltoallv(send_data, send_counts_, send_offsets_, MpiTypeOf<T>::type,
				recv_data, recv_counts_, recv_offsets_, MpiTypeOf<T>::type, comm_);
#if NETWORK_PROBLEM_AYALISYS
		if(mpi.isMaster()) fprintf(IMD_OUT, "OK\n");
#endif
		return recv_data;
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

//-------------------------------------------------------------//
// MPI helper
//-------------------------------------------------------------//

namespace MpiCol {

template <typename T>
int allgatherv(T* sendbuf, T* recvbuf, int sendcount, MPI_Comm comm, int comm_size) {
	int recv_off[comm_size+1], recv_cnt[comm_size];
	MPI_Allgather(&sendcount, 1, MPI_INT, recv_cnt, 1, MPI_INT, comm);
	recv_off[0] = 0;
	for(int i = 0; i < comm_size; ++i) {
		recv_off[i+1] += recv_off[i] + recv_cnt[i];
	}
	MPI_Allgatherv(sendbuf, sendcount, MpiTypeOf<T>::type,
			recvbuf, recv_cnt, recv_off, MpiTypeOf<T>::type, comm);
	return recv_off[comm_size];
}

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

#pragma omp master
		{ scatter.sum(); } // #pragma omp master
#pragma omp barrier
		;
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

#pragma omp master
		{ scatter.sum(); } // #pragma omp master
#pragma omp barrier
		;
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

//-------------------------------------------------------------//
// For print functions
//-------------------------------------------------------------//

double to_giga(int64_t v) { return v / (1024.0*1024.0*1024.0); }
double to_mega(int64_t v) { return v / (1024.0*1024.0); }
double diff_percent(int64_t v, int64_t sum, int demon) {
	double avg = sum / (double)demon;
	return (v - avg) / avg * 100.0;
}
const char* minimum_type(int64_t max_value) {
	if(     max_value < (int64_t(1) <<  7)) return "int8_t";
	else if(max_value < (int64_t(1) <<  8)) return "uint8_t";
	else if(max_value < (int64_t(1) << 15)) return "int16_t";
	else if(max_value < (int64_t(1) << 16)) return "uint16_t";
	else if(max_value < (int64_t(1) << 31)) return "int32_t";
	else if(max_value < (int64_t(1) << 32)) return "uint32_t";
	else return "int64_t";
}

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

//-------------------------------------------------------------//
// Other functions
//-------------------------------------------------------------//

int64_t get_time_in_microsecond()
{
	struct timeval l;
	gettimeofday(&l, NULL);
	return ((int64_t)l.tv_sec*1000000 + l.tv_usec);
}

template <int width> size_t roundup(size_t size)
{
	return (size + width - 1) / width * width;
}

template <int width> size_t get_blocks(size_t size)
{
	return (size + width - 1) / width;
}

inline size_t roundup_2n(size_t size, size_t width)
{
	return (size + width - 1) & -width;
}

inline size_t get_blocks(size_t size, size_t width)
{
	return (size + width - 1) / width;
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
void get_partition(int64_t size, T* sorted, int64_t min_value, int64_t max_value,
		int64_t min_blk_size, int num_part, int part_idx, T*& begin, T*& end)
{
	T blk_size = std::max(min_blk_size, (max_value - min_value + num_part - 1) / num_part);
	int64_t begin_value = min_value + blk_size * part_idx;
	int64_t end_value = begin_value + blk_size;
	begin = std::lower_bound(sorted, sorted + size, begin_value);
	end = std::lower_bound(sorted, sorted + size, end_value);
}

//-------------------------------------------------------------//
// VarInt Encoding
//-------------------------------------------------------------//

namespace vlq {

enum CODING_ENUM {
	MAX_CODE_LENGTH_32 = 5,
	MAX_CODE_LENGTH_64 = 9,
};

#define VARINT_ENCODE_MACRO_32(p, v, l) \
if(v < 128) { \
	p[0] = (uint8_t)v; \
	l = 1; \
} \
else if(v < 128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7); \
	l = 2; \
} \
else if(v < 128*128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14); \
	l = 3; \
} \
else if(v < 128*128*128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21); \
	l = 4; \
} \
else { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28); \
	l = 5; \
}

#define VARINT_ENCODE_MACRO_64(p, v, l) \
if(v < 128) { \
	p[0] = (uint8_t)v; \
	l = 1; \
} \
else if(v < 128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7); \
	l = 2; \
} \
else if(v < 128*128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14); \
	l = 3; \
} \
else if(v < 128*128*128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21); \
	l = 4; \
} \
else if(v < 128LL*128*128*128*128){ \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28); \
	l = 5; \
} \
else if(v < 128LL*128*128*128*128*128){ \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28) | 0x80; \
	p[5]= (uint8_t)(v >> 35); \
	l = 6; \
} \
else if(v < 128LL*128*128*128*128*128*128){ \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28) | 0x80; \
	p[5]= (uint8_t)(v >> 35) | 0x80; \
	p[6]= (uint8_t)(v >> 42); \
	l = 7; \
} \
else if(v < 128LL*128*128*128*128*128*128*128){ \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28) | 0x80; \
	p[5]= (uint8_t)(v >> 35) | 0x80; \
	p[6]= (uint8_t)(v >> 42) | 0x80; \
	p[7]= (uint8_t)(v >> 49); \
	l = 8; \
} \
else { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28) | 0x80; \
	p[5]= (uint8_t)(v >> 35) | 0x80; \
	p[6]= (uint8_t)(v >> 42) | 0x80; \
	p[6]= (uint8_t)(v >> 49) | 0x80; \
	p[8]= (uint8_t)(v >> 56); \
	l = 9; \
}

#define VARINT_DECODE_MACRO_32(p, v, l) \
if(p[0] < 128) { \
	v = p[0]; \
	l = 1; \
} \
else if(p[1] < 128) { \
	v = (p[0] & 0x7F) | ((uint32_t)p[1] << 7); \
	l = 2; \
} \
else if(p[2] < 128) { \
	v = (p[0] & 0x7F) | ((uint32_t)(p[1] & 0x7F) << 7) | \
			((uint32_t)(p[2]) << 14); \
	l = 3; \
} \
else if(p[3] < 128) { \
	v = (p[0] & 0x7F) | ((uint32_t)(p[1] & 0x7F) << 7) | \
			((uint32_t)(p[2] & 0x7F) << 14) | ((uint32_t)(p[3]) << 21); \
	l = 4; \
} \
else { \
	v = (p[0] & 0x7F) | ((uint32_t)(p[1] & 0x7F) << 7) | \
			((uint32_t)(p[2] & 0x7F) << 14) | ((uint32_t)(p[3] & 0x7F) << 21) | \
			((uint32_t)(p[4]) << 28); \
	l = 5; \
}

#define VARINT_DECODE_MACRO_64(p, v, l) \
if(p[0] < 128) { \
	v = p[0]; \
	l = 1; \
} \
else if(p[1] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)p[1] << 7); \
	l = 2; \
} \
else if(p[2] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2]) << 14); \
	l = 3; \
} \
else if(p[3] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3]) << 21); \
	l = 4; \
} \
else if(p[4] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3] & 0x7F) << 21) | \
			((uint64_t)(p[4]) << 28); \
	l = 5; \
} \
else if(p[5] < 128) { \
	v= (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3] & 0x7F) << 21) | \
			((uint64_t)(p[4] & 0x7F) << 28) | ((uint64_t)(p[5]) << 35); \
	l = 6; \
} \
else if(p[6] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3] & 0x7F) << 21) | \
			((uint64_t)(p[4] & 0x7F) << 28) | ((uint64_t)(p[5] & 0x7F) << 35) | \
			((uint64_t)(p[6]) << 42); \
	l = 7; \
} \
else if(p[7] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3] & 0x7F) << 21) | \
			((uint64_t)(p[4] & 0x7F) << 28) | ((uint64_t)(p[5] & 0x7F) << 35) | \
			((uint64_t)(p[6] & 0x7F) << 42) | ((uint64_t)(p[7]) << 49); \
	l = 8; \
} \
else { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3] & 0x7F) << 21) | \
			((uint64_t)(p[4] & 0x7F) << 28) | ((uint64_t)(p[5] & 0x7F) << 35) | \
			((uint64_t)(p[6] & 0x7F) << 42) | ((uint64_t)(p[7] & 0x7F) << 49) | \
			((uint64_t)(p[8]) << 56); \
	l = 9; \
}

int encode(const uint32_t* input, int length, uint8_t* output)
{
	uint8_t* p = output;
	for(int k = 0; k < length; ++k) {
		uint32_t v = input[k];
		int len;
		VARINT_ENCODE_MACRO_32(p, v, len);
		p += len;
	}
	return p - output;
}

int encode(const uint64_t* input, int length, uint8_t* output)
{
	uint8_t* p = output;
	for(int k = 0; k < length; ++k) {
		uint64_t v = input[k];
		int len;
		VARINT_ENCODE_MACRO_64(p, v, len);
		p += len;
	}
	return p - output;
}

int encode_signed(const uint64_t* input, int length, uint8_t* output)
{
	uint8_t* p = output;
	for(int k = 0; k < length; ++k) {
		uint64_t v = input[k];
		v = (v << 1) ^ (((int64_t)v) >> 63);
		int len;
		VARINT_ENCODE_MACRO_64(p, v, len);
		p += len;
	}
	return p - output;
}

int decode(const uint8_t* input, int length, uint32_t* output)
{
	const uint8_t* p = input;
	const uint8_t* p_end = input + length;
	int n = 0;
	for(; p < p_end; ++n) {
		uint32_t v;
		int len;
		VARINT_DECODE_MACRO_32(p, v, len);
		output[n] = v;
		p += len;
	}
	return n;
}

int decode(const uint8_t* input, int length, uint64_t* output)
{
	const uint8_t* p = input;
	const uint8_t* p_end = input + length;
	int n = 0;
	for(; p < p_end; ++n) {
		uint64_t v;
		int len;
		VARINT_DECODE_MACRO_64(p, v, len);
		output[n] = v;
		p += len;
	}
	return n;
}

int decode_signed(const uint8_t* input, int length, uint64_t* output)
{
	const uint8_t* p = input;
	const uint8_t* p_end = input + length;
	int n = 0;
	for(; p < p_end; ++n) {
		uint64_t v;
		int len;
		VARINT_DECODE_MACRO_64(p, v, len);
		output[n] = (v >> 1) ^ (((int64_t)(v << 63)) >> 63);
		p += len;
	}
	return n;
}

int encode_gpu_compat(const uint32_t* input, int length, uint8_t* output)
{
	enum { MAX_CODE_LENGTH = MAX_CODE_LENGTH_32, SIMD_WIDTH = 32 };
	uint8_t tmp_buffer[SIMD_WIDTH][MAX_CODE_LENGTH];
	uint8_t code_length[SIMD_WIDTH];
	int count[MAX_CODE_LENGTH + 1];

	uint8_t* out_ptr = output;
	for(int i = 0; i < length; i += SIMD_WIDTH) {
		int width = std::min(length - i, (int)SIMD_WIDTH);

		for(int k = 0; k < MAX_CODE_LENGTH; ++k) {
			count[k] = 0;
		}
		count[MAX_CODE_LENGTH] = 0;

		for(int k = 0; k < width; ++k) {
			uint32_t v = input[i + k];
			uint8_t* dst = tmp_buffer[k];
			int len;
			VARINT_ENCODE_MACRO_32(dst, v, len);
			code_length[k] = len;
			for(int r = 0; r < len; ++r) {
				++count[r + 1];
			}
		}

		for(int k = 1; k < MAX_CODE_LENGTH; ++k) count[k + 1] += count[k];

		for(int k = 0; k < width; ++k) {
			for(int r = 0; r < code_length[k]; ++r) {
				out_ptr[count[r]++] = tmp_buffer[k][r];
			}
		}

		out_ptr += count[MAX_CODE_LENGTH];
	}

	return out_ptr - output;
}

int encode_gpu_compat(const uint64_t* input, int length, uint8_t* output)
{
	enum { MAX_CODE_LENGTH = MAX_CODE_LENGTH_64, SIMD_WIDTH = 32 };
	uint8_t tmp_buffer[SIMD_WIDTH][MAX_CODE_LENGTH];
	uint8_t code_length[SIMD_WIDTH];
	int count[MAX_CODE_LENGTH + 1];

	uint8_t* out_ptr = output;
	for(int i = 0; i < length; i += SIMD_WIDTH) {
		int width = std::min(length - i, (int)SIMD_WIDTH);

		for(int k = 0; k < MAX_CODE_LENGTH; ++k) {
			count[k] = 0;
		}
		count[MAX_CODE_LENGTH] = 0;

		for(int k = 0; k < width; ++k) {
			uint64_t v = input[i + k];
			uint8_t* dst = tmp_buffer[k];
			int len;
			VARINT_ENCODE_MACRO_64(dst, v, len);
			code_length[k] = len;
			for(int r = 0; r < len; ++r) {
				++count[r + 1];
			}
		}

		for(int k = 1; k < MAX_CODE_LENGTH; ++k) count[k + 1] += count[k];

		for(int k = 0; k < width; ++k) {
			for(int r = 0; r < code_length[k]; ++r) {
				out_ptr[count[r]++] = tmp_buffer[k][r];
			}
		}

		out_ptr += count[MAX_CODE_LENGTH];
	}

	return out_ptr - output;
}

int encode_gpu_compat_signed(const int64_t* input, int length, uint8_t* output)
{
	enum { MAX_CODE_LENGTH = MAX_CODE_LENGTH_64, SIMD_WIDTH = 32 };
	uint8_t tmp_buffer[SIMD_WIDTH][MAX_CODE_LENGTH];
	uint8_t code_length[SIMD_WIDTH];
	int count[MAX_CODE_LENGTH + 1];

	uint8_t* out_ptr = output;
	for(int i = 0; i < length; i += SIMD_WIDTH) {
		int width = std::min(length - i, (int)SIMD_WIDTH);

		for(int k = 0; k < MAX_CODE_LENGTH; ++k) {
			count[k] = 0;
		}
		count[MAX_CODE_LENGTH] = 0;

		for(int k = 0; k < width; ++k) {
			int64_t v_raw = input[i + k];
			uint64_t v = (v_raw < 0) ? ((uint64_t)(~v_raw) << 1) | 1 : ((uint64_t)v_raw << 1);
		//	uint64_t v = (v_raw << 1) ^ (v_raw >> 63);
			assert ((int64_t)v >= 0);
			uint8_t* dst = tmp_buffer[k];
			int len;
			VARINT_ENCODE_MACRO_64(dst, v, len);
			code_length[k] = len;
			for(int r = 0; r < len; ++r) {
				++count[r + 1];
			}
		}

		for(int k = 1; k < MAX_CODE_LENGTH; ++k) count[k + 1] += count[k];

		for(int k = 0; k < width; ++k) {
			for(int r = 0; r < code_length[k]; ++r) {
				out_ptr[count[r]++] = tmp_buffer[k][r];
			}
		}

		out_ptr += count[MAX_CODE_LENGTH];
	}

	return out_ptr - output;
}

int sparsity_factor(int64_t range, int64_t num_values)
{
	if(num_values == 0) return 0;
	const double sparsity = (double)range / (double)num_values;
	int scale;
	if(sparsity < 1.0)
		scale = 1;
	else if(sparsity < 128)
		scale = 2;
	else if(sparsity < 128LL*128)
		scale = 3;
	else if(sparsity < 128LL*128*128)
		scale = 4;
	else if(sparsity < 128LL*128*128*128)
		scale = 5;
	else if(sparsity < 128LL*128*128*128*128)
		scale = 6;
	else if(sparsity < 128LL*128*128*128*128*128)
		scale = 7;
	else if(sparsity < 128LL*128*128*128*128*128*128)
		scale = 8;
	else if(sparsity < 128LL*128*128*128*128*128*128*128)
		scale = 9;
	else
		scale = 10;
	return scale;
}

struct PacketIndex {
	uint32_t offset;
	uint16_t length;
	uint16_t num_int;
};

class BitmapEncoder {
public:

	static int calc_max_packet_size(int64_t max_data_size) {
		int max_threads = omp_get_max_threads();
		return ((max_data_size/max_threads) > 32*1024) ? 16*1024 : 256;
	}

	static int64_t calc_capacity_of_values(int64_t bitmap_size, int num_bits_per_word, int64_t max_data_size) {
		int64_t num_bits = bitmap_size * num_bits_per_word;
		int packet_overhead = sizeof(PacketIndex) + MAX_CODE_LENGTH_64;

		int max_packet_size = calc_max_packet_size(max_data_size);
		int packet_min_bytes = max_packet_size - MAX_CODE_LENGTH_64*2;

		int64_t min_data_bytes = num_bits / 8 - max_packet_size * omp_get_max_threads();
		double overhead_factor = 1 + (double)packet_overhead / (double)packet_min_bytes;
		return (min_data_bytes / overhead_factor) - (num_bits / 128);
	}

	/**
	 * bitmap is executed along only one pass
	 * BitmapF::operator (int64_t)
	 * BitmapF::BitsPerWord
	 * BitmapF::BitmapType
	 */
	template <typename BitmapF, bool b64 = false>
	bool bitmap_to_stream(
			const BitmapF& bitmap, int64_t bitmap_size,
			void* output, int64_t* data_size,
			int64_t max_size)
	{
		typedef BitmapF::BitmapType BitmapType;

		out_len = max_size;
		head = sizeof(uint32_t);
		tail = max_size - sizeof(PacketIndex);
		outbuf = output;

		assert ((data_size % sizeof(uint32_t)) == 0);
		const int max_threads = omp_get_max_threads();
		const int max_packet_size = calc_max_packet_size(max_size);
		const int64_t threshold = max_size;
		bool b_break = false;

#pragma omp parallel reduction(|:b_break)
		{
			uint8_t* buf;
			PacketIndex* pk_idx;
			int remain_packet_length = max_packet_size;
			if(reserve_packet(&buf, &pk_idx, max_packet_size) == false) {
				fprintf(IMD_OUT, "Not enough buffer: bitmap_to_stream\n");
				throw "Not enough buffer: bitmap_to_stream";
			}
			uint8_t* ptr = buf;
			int num_int = 0;

			int64_t chunk_size = (bitmap_size + max_threads - 1) / max_threads;
			int64_t i_start = chunk_size * omp_get_thread_num();
			int64_t i_end = std::min(i_start + chunk_size, bitmap_size);
			int64_t prev_val = 0;

			for(int64_t i = i_start; i < i_end; ++i) {
				BitmapType bmp_val = bitmap(i);
				while(bmp_val != BitmapType(0)) {
					uint32_t bit_idx = __builtin_ctz(bmp_val);
					int64_t new_val = BitmapF::BitsPerWord * i + bit_idx;
					int64_t diff = new_val - prev_val;

					if(remain_packet_length < (b64 ? MAX_CODE_LENGTH_64 : MAX_CODE_LENGTH_32)) {
						pk_idx->length = ptr - buf;
						pk_idx->num_int = num_int;
						if(reserve_packet(&buf, &pk_idx, max_packet_size) == false) {
							b_break = true;
							i = i_end;
							break;
						}
						num_int = 0;
						remain_packet_length = max_packet_size;
					}

					int len;
					if(b64) { VARINT_ENCODE_MACRO_64(ptr, diff, len); }
					else { VARINT_ENCODE_MACRO_32(ptr, diff, len); }
					ptr += len;
					++num_int;

					prev_val = new_val;
					bmp_val &= bmp_val - 1;
				}
			}
		} // #pragma omp parallel reduction(|:b_break)

		if(b_break) {
			*data_size = threshold;
			return false;
		}

		*data_size = compact_output();
		return true;
	}
private:
	int64_t head, tail;
	int64_t out_len;
	uint8_t* outbuf;

	bool reserve_packet(uint8_t** ptr, PacketIndex** pk_idx, int req_size) {
		assert ((req_size % sizeof(uint32_t)) == 0);
		int64_t next_head, next_tail;
#pragma omp critical
		{
			*ptr = outbuf + head;
			next_head = head = head + req_size;
			next_tail = tail = tail - sizeof(PacketIndex);
		}
		*pk_idx = outbuf + next_tail;
		(*pk_idx)->offset = head / sizeof(uint32_t);
		(*pk_idx)->length = 0;
		(*pk_idx)->num_int = 0;
		return next_head <= next_tail;
	}

	int64_t compact_output() {
		int num_packet = (out_len - tail) / sizeof(PacketIndex) - 1;
		PacketIndex* pk_tail = (PacketIndex*)&outbuf[out_len] - 2;
		PacketIndex* pk_head = (PacketIndex*)&outbuf[out_len - tail];
		for(int i = 0; i < (num_packet/2); ++i) {
			std::swap(pk_tail[-i], pk_head[i]);
		}
		pk_tail[1].offset = tail / sizeof(uint32_t); // bann hei

#define O_TO_S(offset) ((offset)*sizeof(uint32_t))
#define L_TO_S(length) roundup<sizeof(uint32_t)>(length)
#define TO_S(offset, length) (O_TO_S(offset) + L_TO_S(length))
		int i = 0;
		for( ; i < num_packet; ++i) {
			// When the empty region length is larger than 32 bytes, break.
			if(O_TO_S(pk_head[i+1].offset - pk_head[i].offset) - L_TO_S(pk_head[i].length) > 32)
				break;
		}
#if VERVOSE_MODE
		fprintf(IMD_OUT, "Move %ld length\n", out_len - sizeof(PacketIndex) - O_TO_S(pk_head[i+1].offset));
#endif
		for( ; i < num_packet; ++i) {
			memmove(outbuf + TO_S(pk_head[i].offset, pk_head[i].length),
					outbuf + O_TO_S(pk_head[i+1].offset),
					(i+1 < num_packet) ? L_TO_S(pk_head[i+1].length) : num_packet*sizeof(PacketIndex));
			pk_head[i+1].offset = pk_head[i].offset + L_TO_S(pk_head[i].length) / sizeof(uint32_t);
		}

		*(uint32_t*)outbuf = pk_head[num_packet].offset;
		return O_TO_S(pk_head[num_packet].offset) + num_packet*sizeof(PacketIndex);
#undef O_TO_S
#undef L_TO_S
#undef TO_S
	}

}; // class BitmapEncoder

template <typename Callback, bool b64 = false>
void decode_stream(void* stream, int64_t data_size, Callback cb) {
	assert (data_size >= 4);
	uint8_t* srcbuf = (uint8_t*)stream;
	uint32_t packet_index_start = *(uint32_t*)srcbuf;
	int64_t pk_offset = packet_index_start * sizeof(uint32_t);
	int num_packets = (data_size - pk_offset) / sizeof(PacketIndex);
	PacketIndex* pk_head = (PacketIndex*)(srcbuf + pk_offset);

	for(int i = 0; i < num_packets; ++i) {
		uint8_t* ptr = srcbuf + pk_head[i].offset * sizeof(uint32_t);
		int num_int = pk_head[i].num_int;
		for(int c = 0; c < num_int; ++c) {
			int len;
			if(b64) { int64_t v; VARINT_DECODE_MACRO_64(ptr, v, len); cb(v); }
			else { int32_t v; VARINT_DECODE_MACRO_32(ptr, v, len); cb(v); }
			ptr += len;
		}
		assert (ptr == srcbuf + pk_head[i].length + pk_head[i].offset * sizeof(uint32_t));
	}
}

} // namespace vlq {

namespace memory {

template <typename T>
class Pool {
public:
	Pool()
	{
	}
	virtual ~Pool() {
		clear();
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

	bool empty() const {
		return free_list_.size() == 0;
	}

	size_t size() const {
		return free_list_.size();
	}

	void clear() {
		for(int i = 0; i < (int)free_list_.size(); ++i) {
			free_list_[i]->~T();
			::free(free_list_[i]);
		}
		free_list_.clear();
	}

protected:
	std::vector<T*> free_list_;

	virtual T* allocate_new() {
		return new (malloc(sizeof(T))) T();
	}
};

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

	/*
	bool empty() const { return super_::empty(); }
	size_t size() const { return super_::size(); }
	void clear() { super_::clear(); }
	*/
protected:
	pthread_mutex_t thread_sync_;
};

template <typename T>
class vector_w : public std::vector<T*>
{
	typedef std::vector<T*> super_;
public:
	~vector_w() {
		for(typename super_::iterator it = this->begin(); it != this->end(); ++it) {
			(*it)->~T();
			::free(*it);
		}
		super_::clear();
	}
};

template <typename T>
class deque_w : public std::deque<T*>
{
	typedef std::deque<T*> super_;
public:
	~deque_w() {
		for(typename super_::iterator it = this->begin(); it != this->end(); ++it) {
			(*it)->~T();
			::free(*it);
		}
		super_::clear();
	}
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

namespace profiling {

class ProfilingInformationStore {
public:
	void submit(double span, const char* content, int number) {
#pragma omp critical (pis_submit_time)
		times_.push_back(TimeElement(span, content, number));
	}
	void submit(int64_t span_micro, const char* content, int number) {
#pragma omp critical (pis_submit_time)
		times_.push_back(TimeElement((double)span_micro / 1000000.0, content, number));
	}
	void submitCounter(int64_t counter, const char* content, int number) {
#pragma omp critical (pis_submit_counter)
		counters_.push_back(CountElement(counter, content, number));
	}
	void reset() {
		times_.clear();
		counters_.clear();
	}
	void printResult() {
		printTimeResult();
		printCountResult();
	}
private:
	struct TimeElement {
		double span;
		const char* content;
		int number;

		TimeElement(double span__, const char* content__, int number__)
			: span(span__), content(content__), number(number__) { }
	};
	struct CountElement {
		int64_t count;
		const char* content;
		int number;

		CountElement(int64_t count__, const char* content__, int number__)
			: count(count__), content(content__), number(number__) { }
	};

	void printTimeResult() {
		int num_times = times_.size();
		double *dbl_times = new double[num_times];
		double *sum_times = new double[num_times];
		double *max_times = new double[num_times];

		for(int i = 0; i < num_times; ++i) {
			dbl_times[i] = times_[i].span;
		}

		MPI_Reduce(dbl_times, sum_times, num_times, MPI_DOUBLE, MPI_SUM, 0, mpi.comm_2d);
		MPI_Reduce(dbl_times, max_times, num_times, MPI_DOUBLE, MPI_MAX, 0, mpi.comm_2d);

		if(mpi.isMaster()) {
			for(int i = 0; i < num_times; ++i) {
				fprintf(stderr, "Time of %s, %d, Avg, %f, Max, %f, (ms)\n", times_[i].content,
						times_[i].number,
						sum_times[i] / mpi.size_2d * 1000.0,
						max_times[i] * 1000.0);
			}
		}

		delete [] dbl_times;
		delete [] sum_times;
		delete [] max_times;
	}

	double displayValue(int64_t value) {
		if(value < int64_t(1000))
			return (double)value;
		else if(value < int64_t(1000)*1000)
			return value / 1000.0;
		else if(value < int64_t(1000)*1000*1000)
			return value / (1000.0*1000);
		else if(value < int64_t(1000)*1000*1000*1000)
			return value / (1000.0*1000*1000);
		else
			return value / (1000.0*1000*1000*1000);
	}

	const char* displaySuffix(int64_t value) {
		if(value < int64_t(1000))
			return "";
		else if(value < int64_t(1000)*1000)
			return "K";
		else if(value < int64_t(1000)*1000*1000)
			return "M";
		else if(value < int64_t(1000)*1000*1000*1000)
			return "G";
		else
			return "T";
	}

	void printCountResult() {
		int num_times = counters_.size();
		int64_t *dbl_times = new int64_t[num_times];
		int64_t *sum_times = new int64_t[num_times];
		int64_t *max_times = new int64_t[num_times];

		for(int i = 0; i < num_times; ++i) {
			dbl_times[i] = counters_[i].count;
		}

		MPI_Reduce(dbl_times, sum_times, num_times, MPI_INT64_T, MPI_SUM, 0, mpi.comm_2d);
		MPI_Reduce(dbl_times, max_times, num_times, MPI_INT64_T, MPI_MAX, 0, mpi.comm_2d);

		if(mpi.isMaster()) {
			for(int i = 0; i < num_times; ++i) {
				int64_t sum = sum_times[i], avg = sum_times[i] / mpi.size, maximum = max_times[i];
				fprintf(stderr, "%s, %d, Sum, %ld, Avg, %ld, Max, %ld\n", counters_[i].content,
						counters_[i].number, sum, avg, maximum);
			}
		}

		delete [] dbl_times;
		delete [] sum_times;
		delete [] max_times;
	}

	std::vector<TimeElement> times_;
	std::vector<CountElement> counters_;
};

ProfilingInformationStore g_pis;

class TimeKeeper {
public:
	TimeKeeper() : start_(get_time_in_microsecond()){ }
	void submit(const char* content, int number) {
		int64_t end = get_time_in_microsecond();
		g_pis.submit(end - start_, content, number);
		start_ = end;
	}
	int64_t getSpanAndReset() {
		int64_t end = get_time_in_microsecond();
		int64_t span = end - start_;
		start_ = end;
		return span;
	}
private:
	int64_t start_;
};

class TimeSpan {
public:
	TimeSpan() : span_(0) { }
	TimeSpan(TimeKeeper& keeper) : span_(keeper.getSpanAndReset()) { }

	void reset() { span_ = 0; }
	TimeSpan& operator += (TimeKeeper& keeper) {
		__sync_fetch_and_add(&span_, keeper.getSpanAndReset());
		return *this;
	}
	TimeSpan& operator -= (TimeKeeper& keeper) {
		__sync_fetch_and_add(&span_, - keeper.getSpanAndReset());
		return *this;
	}
	TimeSpan& operator += (TimeSpan& span) {
		__sync_fetch_and_add(&span_, span.span_);
		return *this;
	}
	TimeSpan& operator -= (TimeSpan& span) {
		__sync_fetch_and_add(&span_, - span.span_);
		return *this;
	}
	TimeSpan& operator += (int64_t span) {
		__sync_fetch_and_add(&span_, span);
		return *this;
	}
	TimeSpan& operator -= (int64_t span) {
		__sync_fetch_and_add(&span_, - span);
		return *this;
	}
	void submit(const char* content, int number) {
		g_pis.submit(span_, content, number);
		span_ = 0;
	}
	double getSpan() {
		return (double)span_ / 1000000.0;
	}
private:
	int64_t span_;
};

} // namespace profiling

#if VERVOSE_MODE
volatile int64_t g_tp_comm;
volatile int64_t g_bu_pred_comm;
volatile int64_t g_bu_bitmap_comm;
volatile int64_t g_bu_list_comm;
volatile int64_t g_expand_bitmap_comm;
volatile int64_t g_expand_list_comm;
volatile double g_gpu_busy_time;
#endif

/* edgefactor = 16, seed1 = 2, seed2 = 3 */
int64_t pf_nedge[] = {
	-1,
	32, // 1
	64,
	128,
	256,
	512,
	1024,
	2048,
	4096 , // 8
	8192 ,
	16383 ,
	32767 ,
	65535 ,
	131070 ,
	262144 ,
	524285 ,
	1048570 ,
	2097137 ,
	4194250 ,
	8388513 ,
	16776976 ,
	33553998 ,
	67108130 ,
	134216177 ,
	268432547 ,
	536865258 ,
	1073731075 ,
	2147462776 ,
	4294927670 ,
	8589858508 ,
	17179724952 ,
	34359466407 ,
	68718955183 , // = 2^36 - 521553
	137437972330, // 33
	274876029861, // 34
	549752273512, // 35
	1099505021204, // 36
	0, // 37
	0 // 38
};

#endif /* UTILS_IMPL_HPP_ */
