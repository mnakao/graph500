#ifndef BFS_HPP_
#define BFS_HPP_
#include <pthread.h>
#include <deque>
#include "utils.hpp"
#include "abstract_comm.hpp"

#define debug(...) debug_print(BFSMN, __VA_ARGS__)

struct LocalPacket {
  enum {
	TOP_DOWN_LENGTH = PRM::PACKET_LENGTH/sizeof(uint32_t),
	BOTTOM_UP_LENGTH = PRM::PACKET_LENGTH/sizeof(int64_t)
  };
  int length;
  int64_t src;
  union {
	uint32_t t[TOP_DOWN_LENGTH];
	int64_t b[BOTTOM_UP_LENGTH];
  } data;
};

class BfsBase
{
	typedef BfsBase ThisType;
	typedef Graph2DCSR GraphType;

public:
	enum {
		ENABLE_WRITING_DEPTH = 1,
		BUCKET_UNIT_SIZE = 1024,
		NBPE = PRM::NBPE,
		LOG_NBPE = PRM::LOG_NBPE,
		NBPE_MASK = PRM::NBPE_MASK,
		BFELL_SORT = PRM::BFELL_SORT,
		LOG_BFELL_SORT = PRM::LOG_BFELL_SORT,
		BFELL_SORT_MASK = PRM::BFELL_SORT_MASK,
		BFELL_SORT_IN_BMP = BFELL_SORT / NBPE,
		BU_SUBSTEP = PRM::NUM_BOTTOM_UP_STREAMS,
	};

	class QueuedVertexes {
	public:
		TwodVertex v[BUCKET_UNIT_SIZE];
		int length;
		enum { SIZE = BUCKET_UNIT_SIZE };

		QueuedVertexes() : length(0) { }
		bool append(TwodVertex val) {
			if(length == SIZE) return false;
			v[length++] = val; return true;
		}
		void append_nocheck(TwodVertex val) {
			v[length++] = val;
		}
		void append_nocheck(TwodVertex pred, TwodVertex tgt) {
			v[length+0] = pred;
			v[length+1] = tgt;
			length += 2;
		}
		bool full() { return (length == SIZE); }
		int size() { return length; }
		void clear() { length = 0; }
	};

	struct ThreadLocalBuffer {
		QueuedVertexes* cur_buffer;
		LocalPacket fold_packet[1];
	};

	BfsBase(){}
	virtual ~BfsBase(){}

	template <typename EdgeList>
	void construct(EdgeList* edge_list)
	{
		int log_local_verts_unit = get_msb_index(std::max<int>(BFELL_SORT, NBPE) * 8);
		detail::GraphConstructor2DCSR<EdgeList> constructor;
		constructor.construct(edge_list, log_local_verts_unit, graph_);
	}

	GraphType graph_;

	int64_t get_bitmap_size_src() const {
		return graph_.num_local_verts_ / NBPE * mpi.size_2dr;
	}
	int64_t get_bitmap_size_tgt() const {
		return graph_.num_local_verts_ / NBPE * mpi.size_2dc;
	}
	int64_t get_bitmap_size_local() const {
		return graph_.num_local_verts_ / NBPE;
	}

	template <typename T>
	void get_shared_mem_pointer(void*& ptr, int64_t width, T** local_ptr, T** orig_ptr) {
		if(orig_ptr) *orig_ptr = (T*)ptr;
		if(local_ptr) *local_ptr = (T*)ptr + width*mpi.rank_z;
		ptr = (uint8_t*)ptr + width*sizeof(T)*mpi.size_z;
	}

	class CommBufferPool {
	public:
		void allocate_memory(int size) {
			first_buffer_ = cache_aligned_xmalloc(size);
			second_buffer_ = cache_aligned_xmalloc(size);
			current_index_ = 0;
			pool_buffer_size_ = size;
			num_buffers_ = size / PRM::COMM_BUFFER_SIZE;
		}

		void deallocate_memory() {
			free(first_buffer_); first_buffer_ = NULL;
			free(second_buffer_); second_buffer_ = NULL;
		}

		void* get_next() {
			int idx = current_index_++;
			if(num_buffers_ <= idx) {
				fprintf(IMD_OUT, "num_buffers_ <= idx (num_buffers=%d)\n", num_buffers_);
				throw "Error: buffer size not enough";
			}
			return (uint8_t*)first_buffer_ + PRM::COMM_BUFFER_SIZE * idx;
		}

		void* clear_buffers() {
			current_index_ = 0;
			return first_buffer_;
		}

		void* second_buffer() {
			return second_buffer_;
		}

		int pool_buffer_size() {
			return pool_buffer_size_;
		}

	protected:
		int pool_buffer_size_;
		void* first_buffer_;
		void* second_buffer_;
		int current_index_;
		int num_buffers_;
	};

	struct TopDownRow {
		int64_t src;
		int length;
		uint32_t* ptr;
	};

	CommBufferPool a2a_comm_buf_;
	ThreadLocalBuffer** thread_local_buffer_;
	memory::ConcurrentPool<QueuedVertexes> nq_empty_buffer_;
	memory::ConcurrentStack<QueuedVertexes*> nq_;

  // switch parameters
  double denom_to_bottom_up_; // alpha
  double denom_bitmap_to_list_; // gamma

  // cq_list_ is a pointer to work_buf_ or work_extra_buf_
  TwodVertex* cq_list_;
  TwodVertex cq_size_;
  void* new_visited_; // shared memory but point to the local portion
  void* old_visited_; // shared memory but point to the local portion
  BitmapType* shared_visited_; // shared memory
  int64_t* pred_; // passed from main method
  int current_level_;
  bool growing_or_shrinking_;
  bool packet_buffer_is_dirty_;
};

#undef debug

#endif /* BFS_HPP_ */

