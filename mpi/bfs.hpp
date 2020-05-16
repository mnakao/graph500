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
		// Number of CQ bitmap entries represent as 1 bit in summary.
		// Since the type of bitmap entry is int32_t and 1 cache line is composed of 32 bitmap entries,
		// 32 is effective value.
		ENABLE_WRITING_DEPTH = 1,

		BUCKET_UNIT_SIZE = 1024,

		// non-parameters
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

	BfsBase()
	  : top_down_comm_(this)
		, bottom_up_comm_(this)
		, td_comm_(mpi.comm_2dc, &top_down_comm_)
		, bu_comm_(mpi.comm_2dr, &bottom_up_comm_)
		, denom_to_bottom_up_(DENOM_TOPDOWN_TO_BOTTOMUP)
		, denom_bitmap_to_list_(DENOM_BITMAP_TO_LIST)
	{
	}

	virtual ~BfsBase()
	{
	}

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

	template <typename T>
	class CommHandlerBase : public AlltoallBufferHandler {
	public:
		enum { BUF_SIZE = PRM::COMM_BUFFER_SIZE / sizeof(T) };
		CommHandlerBase(ThisType* this__)
			: this_(this__)
			, pool_(&this__->a2a_comm_buf_)
		{ }
		virtual ~CommHandlerBase() { }
		virtual void* get_buffer() {
			return this->pool_->get_next();
		}
		virtual void add(void* buffer, void* ptr__, int offset, int length) {
			assert (offset >= 0);
			assert (offset + length <= BUF_SIZE);
			memcpy((T*)buffer + offset, ptr__, length*sizeof(T));
		}
		virtual void* clear_buffers() {
			return this->pool_->clear_buffers();
		}
		virtual void* second_buffer() {
			return this->pool_->second_buffer();
		}
		virtual int max_size() {
			return this->pool_->pool_buffer_size();
		}
		virtual int buffer_length() {
			return BUF_SIZE;
		}
		virtual MPI_Datatype data_type() {
			return MpiTypeOf<T>::type;
		}
		virtual int element_size() {
			return sizeof(T);
		}
		virtual void finish() { }
	protected:
		ThisType* this_;
		CommBufferPool* pool_;
	};

	struct TopDownRow {
		int64_t src;
		int length;
		uint32_t* ptr;
	};

	class TopDownCommHandler : public CommHandlerBase<uint32_t> {
	public:
		TopDownCommHandler(ThisType* this__)
			: CommHandlerBase<uint32_t>(this__)
			, tmp_rows(NULL)
			, max_num_rows(0)
			, num_rows(0)
			  { }

		~TopDownCommHandler() {
			if(tmp_rows != NULL) { free(tmp_rows); tmp_rows = NULL; }
		}

		virtual void received(void* buf, int offset, int length, int src) {
			if(this_->growing_or_shrinking_) {
				this->this_->top_down_receive<true>((uint32_t*)buf + offset, length, tmp_rows, &num_rows);
			}
			else {
				this->this_->top_down_receive<false>((uint32_t*)buf + offset, length, tmp_rows, &num_rows);
			}
			assert (num_rows < max_num_rows);
		}

		virtual void finish() {
			if(num_rows == 0) return ;
			if(num_rows > max_num_rows) {
				fprintf(IMD_OUT, "Insufficient temporary rows buffer\n");
				throw "Insufficient temporary rows buffer";
			}
			if(this_->growing_or_shrinking_) {
				this->this_->top_down_row_receive<true>(tmp_rows, num_rows);
			}
			else {
				this->this_->top_down_row_receive<false>(tmp_rows, num_rows);
			}
			num_rows = 0;
		}

		TopDownRow* tmp_rows;
		int max_num_rows;
		volatile int num_rows;
	};

	class BottomUpCommHandler : public CommHandlerBase<int64_t> {
	public:
		BottomUpCommHandler(ThisType* this__)
			: CommHandlerBase<int64_t>(this__)
			  { }

		virtual void received(void* buf, int offset, int length, int src) {
		}
	};

	template <bool growing>
	void top_down_row_receive(TopDownRow* rows, int num_rows) {
		int num_threads = omp_get_max_threads();
		int num_splits = num_threads * 8;
		// process from tail because computation cost is higher in tail
		volatile int procces_counter = num_splits - 1;
		//volatile int procces_counter = 0;

#pragma omp parallel
		{
			int tid = omp_get_thread_num();
			ThreadLocalBuffer* tlb = thread_local_buffer_[tid];
			QueuedVertexes* buf = tlb->cur_buffer;
			if(buf == NULL) buf = nq_empty_buffer_.get();
			BitmapType* visited = (BitmapType*)new_visited_;
			int64_t* restrict const pred = pred_;
			LocalVertex* invert_map = graph_.invert_map_;

			// for id converter //
			int lgl = graph_.local_bits_;
			LocalVertex lmask = (LocalVertex(1) << lgl) - 1;
			// ------------------- //

			while(true) {
				int split = __sync_fetch_and_add(&procces_counter, -1);
				if(split < 0) break;
				//int split = __sync_fetch_and_add(&procces_counter, 1);
				//if(split >= num_splits) break;

				for(int r = 0; r < num_rows; ++r) {
					uint32_t* ptr = rows[r].ptr;
					int length = rows[r].length;
					int64_t pred_v = rows[r].src | (int64_t(current_level_) << 48);

					int width_per_split = (length + num_splits - 1) / num_splits;
					int off_start = std::min(length, width_per_split * split);
					int off_end = std::min(length, off_start + width_per_split);

					for(int i = off_start; i < off_end; ++i) {
						LocalVertex tgt_local = ptr[i] & lmask;
						if(growing) {
							// TODO: which is better ?
							//LocalVertex tgt_orig = invert_map[tgt_local];
							const TwodVertex word_idx = tgt_local >> LOG_NBPE;
							const int bit_idx = tgt_local & NBPE_MASK;
							const BitmapType mask = BitmapType(1) << bit_idx;
							if((visited[word_idx] & mask) == 0) { // if this vertex has not visited
								if((__sync_fetch_and_or(&visited[word_idx], mask) & mask) == 0) {
									LocalVertex tgt_orig = invert_map[tgt_local];
									assert (pred[tgt_orig] == -1);
									pred[tgt_orig] = pred_v;
									if(buf->full()) {
										nq_.push(buf); buf = nq_empty_buffer_.get();
									}
									buf->append_nocheck(tgt_local);
								}
							}
						}
						else {
							LocalVertex tgt_orig = invert_map[tgt_local];
							if(pred[tgt_orig] == -1) {
								if(__sync_bool_compare_and_swap(&pred[tgt_orig], -1, pred_v)) {
									if(buf->full()) {
										nq_.push(buf); buf = nq_empty_buffer_.get();
									}
									buf->append_nocheck(tgt_local);
								}
							}
						}
					}
				}
			}
			tlb->cur_buffer = buf;
		} // #pragma omp parallel
	}

	template <bool growing>
	void top_down_receive(uint32_t* stream, int length, TopDownRow* rows, volatile int* num_rows) {
		ThreadLocalBuffer* tlb = thread_local_buffer_[omp_get_thread_num()];
		QueuedVertexes* buf = tlb->cur_buffer;
		if(buf == NULL) buf = nq_empty_buffer_.get();
		BitmapType* visited = (BitmapType*)new_visited_;
		int64_t* restrict const pred = pred_;
		const int cur_level = current_level_;
		int64_t pred_v = -1;
		LocalVertex* invert_map = graph_.invert_map_;

		// for id converter //
		int lgl = graph_.local_bits_;
		LocalVertex lmask = (LocalVertex(1) << lgl) - 1;
		// ------------------- //

		for(int i = 0; i < length; ++i) {
			uint32_t v = stream[i];
			if(v & 0x80000000u) {
				int64_t src = (int64_t(v & 0xFFFF) << 32) | stream[i+1];
				pred_v = src | (int64_t(cur_level) << 48);
				if(v & 0x40000000u) {
					int length_i = stream[i+2];
#if TOP_DOWN_RECV_LB
					if(length_i < PRM::TOP_DOWN_PENDING_WIDTH)
#endif // #if TOP_DOWN_RECV_LB
					{
						assert (pred_v != -1);
						for(int c = 0; c < length_i; ++c) {
							LocalVertex tgt_local = stream[i+3+c] & lmask;
							if(growing) {
								// TODO: which is better ?
								//LocalVertex tgt_orig = invert_map[tgt_local];
								const TwodVertex word_idx = tgt_local >> LOG_NBPE;
								const int bit_idx = tgt_local & NBPE_MASK;
								const BitmapType mask = BitmapType(1) << bit_idx;
								if((visited[word_idx] & mask) == 0) { // if this vertex has not visited
									if((__sync_fetch_and_or(&visited[word_idx], mask) & mask) == 0) {
										LocalVertex tgt_orig = invert_map[tgt_local];
										assert (pred[tgt_orig] == -1);
										pred[tgt_orig] = pred_v;
										if(buf->full()) {
											nq_.push(buf); buf = nq_empty_buffer_.get();
										}
										buf->append_nocheck(tgt_local);
									}
								}
							}
							else {
								LocalVertex tgt_orig = invert_map[tgt_local];
								if(pred[tgt_orig] == -1) {
									if(__sync_bool_compare_and_swap(&pred[tgt_orig], -1, pred_v)) {
										if(buf->full()) {
											nq_.push(buf); buf = nq_empty_buffer_.get();
										}
										buf->append_nocheck(tgt_local);
									}
								}
							}
						}
					}
#if TOP_DOWN_RECV_LB
					else {
						int put_off = __sync_fetch_and_add(num_rows, 1);
						rows[put_off].length = length_i;
						rows[put_off].ptr = &stream[i+3];
						rows[put_off].src = src;
					}
#endif // #if TOP_DOWN_RECV_LB

					i += 2 + length_i;
				}
				else {
					i += 1;
				}
			}
			else {
				assert (pred_v != -1);

				LocalVertex tgt_local = v & lmask;
				if(growing) {
					// TODO: which is better ?
					//LocalVertex tgt_orig = invert_map[tgt_local];
					const TwodVertex word_idx = tgt_local >> LOG_NBPE;
					const int bit_idx = tgt_local & NBPE_MASK;
					const BitmapType mask = BitmapType(1) << bit_idx;
					if((visited[word_idx] & mask) == 0) { // if this vertex has not visited
						if((__sync_fetch_and_or(&visited[word_idx], mask) & mask) == 0) {
							LocalVertex tgt_orig = invert_map[tgt_local];
							assert (pred[tgt_orig] == -1);
							pred[tgt_orig] = pred_v;
							if(buf->full()) {
								nq_.push(buf); buf = nq_empty_buffer_.get();
							}
							buf->append_nocheck(tgt_local);
						}
					}
				}
				else {
					LocalVertex tgt_orig = invert_map[tgt_local];
					if(pred[tgt_orig] == -1) {
						if(__sync_bool_compare_and_swap(&pred[tgt_orig], -1, pred_v)) {
							if(buf->full()) {
								nq_.push(buf); buf = nq_empty_buffer_.get();
							}
							buf->append_nocheck(tgt_local);
						}
					}
				}
			}
		}
		tlb->cur_buffer = buf;
	}

	CommBufferPool a2a_comm_buf_;
	TopDownCommHandler top_down_comm_;
	BottomUpCommHandler bottom_up_comm_;
	AsyncAlltoallManager td_comm_;
	AsyncAlltoallManager bu_comm_;
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

