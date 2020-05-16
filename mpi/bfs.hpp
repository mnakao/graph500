#ifndef BFS_HPP_
#define BFS_HPP_
#include <pthread.h>
#include <deque>
#include "utils.hpp"
#include "fiber.hpp"
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

	template <typename T>
	void get_visited_pointers(T** ptrs, int num_ptrs, void* visited_buf, int split_count) {
		int step_bitmap_width = get_bitmap_size_local() / split_count;
		for(int i = 0; i < num_ptrs; ++i) {
			ptrs[i] = (T*)((BitmapType*)visited_buf + step_bitmap_width*i);
		}
	}

	struct NQBitmapCombiner {
		BitmapType* new_visited;
		BitmapType* old_visited;
		NQBitmapCombiner(ThisType* this__)
			: new_visited((BitmapType*)this__->new_visited_)
			, old_visited((BitmapType*)this__->old_visited_) { }
		BitmapType operator ()(int i) { return new_visited[i] & ~(old_visited[i]); }
	};

	void top_down_send(int64_t tgt, int lgl, int r_mask,
			LocalPacket* packet_array, int64_t src
	) {
		int dest = (tgt >> lgl) & r_mask;
		LocalPacket& pk = packet_array[dest];
		if(pk.length > LocalPacket::TOP_DOWN_LENGTH-3) { // low probability
			td_comm_.put(pk.data.t, pk.length, dest);
			pk.src = -1;
			pk.length = 0;
		}
		if(pk.src != src) { // TODO: use conditional branch
			pk.src = src;
			pk.data.t[pk.length++] = (src >> 32) | 0x80000000u;
			pk.data.t[pk.length++] = (uint32_t)src;
		}
		pk.data.t[pk.length++] = tgt & ((uint32_t(1) << lgl) - 1);
	}

	void top_down_send_large(int64_t* edge_array, int64_t start, int64_t end,
			int lgl, int r_mask, int64_t src)
	{
		assert (end > start);
		for(int i = 0; i < mpi.size_2dr; ++i) {
			if(start >= end) break;
			int s_dest = (edge_array[start] >> lgl) & r_mask;
			if(s_dest > i) continue;

			// search the destination change point with binary search
			int64_t left = start;
			int64_t right = end;
			int64_t next = std::min(left + (right - left) / (mpi.size_2dr - i) * 2, end - 1);
			do {
				int dest = (edge_array[next] >> lgl) & r_mask;
				if(dest > i) {
					right = next;
				}
				else {
					left = next;
				}
				next = (left + right) / 2;
			} while(left < next);
			// start ... right -> i
			td_comm_.put_ptr(edge_array + start, right - start, src, i);
			start = right;
		}
		assert(start == end);
	}

	void top_down_parallel_section(bool bitmap_or_list) {
		bool clear_packet_buffer = packet_buffer_is_dirty_;
		packet_buffer_is_dirty_ = false;

#if TOP_DOWN_SEND_LB == 2
#define IF_LARGE_EDGE if(e_end - e_start > PRM::TOP_DOWN_PENDING_WIDTH/10)
#define ELSE else
#else
#define IF_LARGE_EDGE
#define ELSE
#endif

		debug("begin parallel");
#pragma omp parallel
		{
			SET_OMP_AFFINITY;
			//int max_threads = omp_get_num_threads();
			int64_t* edge_array = graph_.edge_array_;
			LocalPacket* packet_array =
					thread_local_buffer_[omp_get_thread_num()]->fold_packet;
			if(clear_packet_buffer) {
				for(int target = 0; target < mpi.size_2dr; ++target) {
					packet_array[target].src = -1;
					packet_array[target].length = 0;
				}
			}
			int lgl = graph_.local_bits_;
			//int vertex_bits = graph_.r_bits_ + lgl;
			int r_mask = (1 << graph_.r_bits_) - 1;
			int P = mpi.size_2d;
			int R = mpi.size_2dr;
			int r = mpi.rank_2dr;
			uint32_t local_mask = (uint32_t(1) << lgl) - 1;
			int64_t L = graph_.num_local_verts_;

			// count total edges
			if(bitmap_or_list) {
				BitmapType* cq_bitmap = shared_visited_;
				int64_t bitmap_size = get_bitmap_size_local() * mpi.size_2dc;
	#pragma omp for
				for(int64_t word_idx = 0; word_idx < bitmap_size; ++word_idx) {
					BitmapType cq_bit_i = cq_bitmap[word_idx];
					if(cq_bit_i == BitmapType(0)) continue;

					BitmapType row_bitmap_i = graph_.row_bitmap_[word_idx];
					BitmapType bit_flags = cq_bit_i & row_bitmap_i;
					TwodVertex bmp_row_sums = graph_.row_sums_[word_idx];
					while(bit_flags != BitmapType(0)) {
						BitmapType cq_bit = bit_flags & (-bit_flags);
						BitmapType low_mask = cq_bit - 1;
						bit_flags &= ~cq_bit;
						int bit_idx = __builtin_popcountl(low_mask);
						TwodVertex compact = word_idx * NBPE + bit_idx;
						TwodVertex src_c = word_idx / get_bitmap_size_local(); // TODO:
						TwodVertex non_zero_off = bmp_row_sums + __builtin_popcountl(row_bitmap_i & low_mask);
						int64_t src_orig =
								int64_t(graph_.orig_vertexes_[non_zero_off]) * P + src_c * R + r;
	#if ISOLATE_FIRST_EDGE
						top_down_send(graph_.isolated_edges_[non_zero_off], lgl,
								r_mask, packet_array, src_orig
							);
	#endif // #if ISOLATE_FIRST_EDGE
						int64_t e_start = graph_.row_starts_[non_zero_off];
						int64_t e_end = graph_.row_starts_[non_zero_off+1];
						IF_LARGE_EDGE
#if TOP_DOWN_SEND_LB > 0
						{
							top_down_send_large(edge_array, e_start, e_end, lgl, r_mask, src_orig);
						}
#endif // #if TOP_DOWN_SEND_LB > 0
						ELSE
#if TOP_DOWN_SEND_LB != 1
						{
							for(int64_t e = e_start; e < e_end; ++e) {
								top_down_send(edge_array[e], lgl,
										r_mask, packet_array, src_orig
									);
							}
						}
#endif // #if TOP_DOWN_SEND_LB != 1
					} // while(bit_flags != BitmapType(0)) {
				} // #pragma omp for // implicit barrier
			}
			else {
				TwodVertex* cq_list = (TwodVertex*)cq_list_;
	#pragma omp for
				for(int64_t i = 0; i < int64_t(cq_size_); ++i) {
					SeparatedId src(cq_list[i]);
					TwodVertex src_c = src.value >> lgl;
					TwodVertex compact = src_c * L + (src.value & local_mask);
					TwodVertex word_idx = compact >> LOG_NBPE;
					int bit_idx = compact & NBPE_MASK;
					BitmapType row_bitmap_i = graph_.row_bitmap_[word_idx];
					BitmapType mask = BitmapType(1) << bit_idx;
					if(row_bitmap_i & mask) {
						BitmapType low_mask = (BitmapType(1) << bit_idx) - 1;
						TwodVertex non_zero_off = graph_.row_sums_[word_idx] +
								__builtin_popcountl(graph_.row_bitmap_[word_idx] & low_mask);
						int64_t src_orig =
								int64_t(graph_.orig_vertexes_[non_zero_off]) * P + src_c * R + r;
	#if ISOLATE_FIRST_EDGE
						top_down_send(graph_.isolated_edges_[non_zero_off], lgl,
								r_mask, packet_array, src_orig
							);
	#endif // #if ISOLATE_FIRST_EDGE
						int64_t e_start = graph_.row_starts_[non_zero_off];
						int64_t e_end = graph_.row_starts_[non_zero_off+1];
						IF_LARGE_EDGE
#if TOP_DOWN_SEND_LB > 0
						{
							top_down_send_large(edge_array, e_start, e_end, lgl, r_mask, src_orig);
						}
#endif // #if TOP_DOWN_SEND_LB > 0
						ELSE
#if TOP_DOWN_SEND_LB != 1
						{
							for(int64_t e = e_start; e < e_end; ++e) {
								top_down_send(edge_array[e], lgl,
										r_mask, packet_array, src_orig
									);
							}
						}
#endif // #if TOP_DOWN_SEND_LB != 1
					} // if(row_bitmap_i & mask) {
				} // #pragma omp for // implicit barrier
			}

			// flush buffer
#pragma omp for
			for(int target = 0; target < mpi.size_2dr; ++target) {
				for(int i = 0; i < omp_get_num_threads(); ++i) {
					LocalPacket* packet_array =
							thread_local_buffer_[i]->fold_packet;
					LocalPacket& pk = packet_array[target];
					if(pk.length > 0) {
						td_comm_.put(pk.data.t, pk.length, target);
						pk.src = -1;
						pk.length = 0;
					}
				}
			} // #pragma omp for
		} // #pragma omp parallel reduction(+:num_edge_relax)
#undef IF_LARGE_EDGE
#undef ELSE
		debug("finished parallel");
	}

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

	//-------------------------------------------------------------//
	// bottom-up search
	//-------------------------------------------------------------//

	void botto_up_print_stt(int64_t num_blocks, int64_t num_vertexes, int* nq_count) {
		int64_t send_stt[2] = { num_vertexes, num_blocks };
		int64_t sum_stt[2];
		int64_t max_stt[2];
		MPI_Reduce(send_stt, sum_stt, 2, MpiTypeOf<int64_t>::type, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(send_stt, max_stt, 2, MpiTypeOf<int64_t>::type, MPI_MAX, 0, MPI_COMM_WORLD);
		if(mpi.isMaster() && sum_stt[0] != 0) {
			print_with_prefix("Bottom-Up using List. Total %f M Vertexes / %f M Blocks = %f Max %f %%+ Vertexes %f %%+ Blocks",
					to_mega(sum_stt[0]), to_mega(sum_stt[1]), to_mega(sum_stt[0]) / to_mega(sum_stt[1]),
					diff_percent(max_stt[0], sum_stt[0], mpi.size_2d),
					diff_percent(max_stt[1], sum_stt[1], mpi.size_2d));
		}
		int count_length = mpi.size_2dc;
		int start_proc = mpi.rank_2dc;
		int size_mask = count_length - 1;
		int64_t phase_count[count_length];
		int64_t phase_recv[count_length];
		for(int i = 0; i < count_length; ++i) {
			phase_count[i] = nq_count[(start_proc + i) & size_mask];
		}
		MPI_Reduce(phase_count, phase_recv, count_length, MpiTypeOf<int64_t>::type, MPI_SUM, 0, MPI_COMM_WORLD);
		if(mpi.isMaster()) {
			int64_t total_nq = 0;
			for(int i = 0; i < count_length; ++i) {
				total_nq += phase_recv[i];
			}
			print_with_prefix("Bottom-Up: %" PRId64 " vertexes found. Break down ...", total_nq);
			for(int i = 0; i < count_length; ++i) {
				print_with_prefix("step %d / %d  %f M Vertexes ( %f %% )",
						i+1, count_length, to_mega(phase_recv[i]), (double)phase_recv[i] / (double)total_nq * 100.0);
			}
		}
	}

	struct UnvisitedBitmapFunctor {
		BitmapType* visited;
		UnvisitedBitmapFunctor(BitmapType* visited__)
			: visited(visited__) { }
		BitmapType operator()(int i) { return ~(visited[i]); }
	};

	void flush_bottom_up_send_buffer(LocalPacket* buffer, int target_rank) {
		int bulk_send_size = BottomUpCommHandler::BUF_SIZE;
		for(int offset = 0; offset < buffer->length; offset += bulk_send_size) {
			int length = std::min(buffer->length - offset, bulk_send_size);
			bu_comm_.put(buffer->data.b + offset, length, target_rank);
		}
		buffer->length = 0;
	}

	void bottom_up_search_bitmap_process_block(
			BitmapType* __restrict__ phase_bitmap,
			int off_start,
			int off_end,
			int phase_bmp_off,
			LocalPacket* buffer)
	{

		int lgl = graph_.local_bits_;
		TwodVertex L = graph_.num_local_verts_;
		int r_bits = graph_.r_bits_;
		int orig_lgl = graph_.orig_local_bits_;

		const BitmapType* __restrict__ row_bitmap = graph_.row_bitmap_;
		const BitmapType* __restrict__ shared_visited = shared_visited_;
		const TwodVertex* __restrict__ row_sums = graph_.row_sums_;
		const int64_t* __restrict__ isolated_edges = graph_.isolated_edges_;
		const int64_t* __restrict__ row_starts = graph_.row_starts_;
		const LocalVertex* __restrict__ orig_vertexes = graph_.orig_vertexes_;
		const int64_t* __restrict__ edge_array = graph_.edge_array_;

		//TwodVertex lmask = (TwodVertex(1) << lgl) - 1;
		int num_send = 0;
#if CONSOLIDATE_IFE_PROC
		for(int64_t blk_bmp_off = off_start; blk_bmp_off < off_end; ++blk_bmp_off) {
			BitmapType row_bmp_i = *(row_bitmap + phase_bmp_off + blk_bmp_off);
			BitmapType visited_i = *(phase_bitmap + blk_bmp_off);
			TwodVertex bmp_row_sums = *(row_sums + phase_bmp_off + blk_bmp_off);
			BitmapType bit_flags = (~visited_i) & row_bmp_i;
			while(bit_flags != BitmapType(0)) {
				BitmapType vis_bit = bit_flags & (-bit_flags);
				BitmapType mask = vis_bit - 1;
				bit_flags &= ~vis_bit;
				int idx = __builtin_popcountl(mask);
				TwodVertex non_zero_idx = bmp_row_sums + __builtin_popcountl(row_bmp_i & mask);
				LocalVertex tgt_orig = orig_vertexes[non_zero_idx];
				// short cut
				int64_t src = isolated_edges[non_zero_idx];
				TwodVertex bit_idx = SeparatedId(SeparatedId(src).low(r_bits + lgl)).compact(lgl, L);
				if(shared_visited[bit_idx >> PRM::LOG_NBPE] & (BitmapType(1) << (bit_idx & PRM::NBPE_MASK))) {
					// add to next queue
					visited_i |= vis_bit;
					buffer->data.b[num_send++] = ((src >> lgl) << orig_lgl) | tgt_orig;
					// end this row
					continue;
				}
				int64_t e_start = row_starts[non_zero_idx];
				int64_t e_end = row_starts[non_zero_idx+1];
				for(int64_t e = e_start; e < e_end; ++e) {
					int64_t src = edge_array[e];
					TwodVertex bit_idx = SeparatedId(SeparatedId(src).low(r_bits + lgl)).compact(lgl, L);
					if(shared_visited[bit_idx >> PRM::LOG_NBPE] & (BitmapType(1) << (bit_idx & PRM::NBPE_MASK))) {
						// add to next queue
						visited_i |= vis_bit;
						buffer->data.b[num_send++] = ((src >> lgl) << orig_lgl) | tgt_orig;
						// end this row
						break;
					}
				}
			} // while(bit_flags != BitmapType(0)) {
			// write back
			*(phase_bitmap + blk_bmp_off) = visited_i;
		} // #pragma omp for

#else // #if CONSOLIDATE_IFE_PROC
		for(int64_t blk_bmp_off = off_start; blk_bmp_off < off_end; ++blk_bmp_off) {
			BitmapType row_bmp_i = *(row_bitmap + phase_bmp_off + blk_bmp_off);
			BitmapType visited_i = *(phase_bitmap + blk_bmp_off);
			TwodVertex bmp_row_sums = *(row_sums + phase_bmp_off + blk_bmp_off);
			BitmapType bit_flags = (~visited_i) & row_bmp_i;
			while(bit_flags != BitmapType(0)) {
				BitmapType vis_bit = bit_flags & (-bit_flags);
				BitmapType mask = vis_bit - 1;
				bit_flags &= ~vis_bit;
				int idx = __builtin_popcountl(mask);
				TwodVertex non_zero_idx = bmp_row_sums + __builtin_popcountl(row_bmp_i & mask);
				// short cut
				TwodVertex separated_src = isolated_edges[non_zero_idx];
				TwodVertex bit_idx = (separated_src >> lgl) * L + (separated_src & lmask);
				if(shared_visited[bit_idx >> PRM::LOG_NBPE] & (BitmapType(1) << (bit_idx & PRM::NBPE_MASK))) {
					// add to next queue
					visited_i |= vis_bit;
					buffer->data.b[num_send+0] = separated_src;
					buffer->data.b[num_send+1] = (phase_bmp_off + blk_bmp_off) * PRM::NBPE + idx;
					num_send += 2;
				}
			} // while(bit_flags != BitmapType(0)) {
			// write back
			*(phase_bitmap + blk_bmp_off) = visited_i;
		} // #pragma omp for

		for(int64_t blk_bmp_off = off_start; blk_bmp_off < off_end; ++blk_bmp_off) {
			BitmapType row_bmp_i = *(row_bitmap + phase_bmp_off + blk_bmp_off);
			BitmapType visited_i = *(phase_bitmap + blk_bmp_off);
			TwodVertex bmp_row_sums = *(row_sums + phase_bmp_off + blk_bmp_off);
			BitmapType bit_flags = (~visited_i) & row_bmp_i;
			while(bit_flags != BitmapType(0)) {
				BitmapType vis_bit = bit_flags & (-bit_flags);
				BitmapType mask = vis_bit - 1;
				bit_flags &= ~vis_bit;
				int idx = __builtin_popcountl(mask);
				TwodVertex non_zero_idx = bmp_row_sums + __builtin_popcountl(row_bmp_i & mask);
				int64_t e_start = row_starts[non_zero_idx];
				int64_t e_end = row_starts[non_zero_idx+1];
				for(int64_t e = e_start; e < e_end; ++e) {
					TwodVertex separated_src = edge_array[e];
					TwodVertex bit_idx = (separated_src >> lgl) * L + (separated_src & lmask);
					if(shared_visited[bit_idx >> PRM::LOG_NBPE] & (BitmapType(1) << (bit_idx & PRM::NBPE_MASK))) {
						// add to next queue
						visited_i |= vis_bit;
						buffer->data.b[num_send+0] = separated_src;
						buffer->data.b[num_send+1] = (phase_bmp_off + blk_bmp_off) * PRM::NBPE + idx;
						num_send += 2;
						// end this row
						break;
					}
				}
			} // while(bit_flags != BitmapType(0)) {
			// write back
			*(phase_bitmap + blk_bmp_off) = visited_i;
		} // #pragma omp for
#endif // #if CONSOLIDATE_IFE_PROC

		buffer->length = num_send;
	}

	void prepare_sssp() { }
	void run_sssp(int64_t root, int64_t* pred) { }
	void end_sssp() { }

	// members
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
	int nq_size_;
	int max_nq_size_;
	int64_t global_nq_size_;

	// size = local bitmap width
	// These two buffer is swapped at the beginning of every backward step
	void* new_visited_; // shared memory but point to the local portion
	void* old_visited_; // shared memory but point to the local portion
	void* visited_buffer_; // shared memory but point to the local portion
	void* visited_buffer_orig_; // shared memory
	// size = 2
	int new_visited_list_size_[BU_SUBSTEP];
	int old_visited_list_size_[BU_SUBSTEP];


	// 1. CQ at the top-down phase
	// 2. NQ receive buffer at the bottom-up expand phase
	// 3. Working memory at the bottom-up search phase
	void* work_buf_; // shared memory but point to the local portion
	void* work_extra_buf_; // for large CQ in the top down phase
	int64_t work_buf_size_; // in bytes
	int64_t work_extra_buf_size_;

	BitmapType* shared_visited_; // shared memory
	TwodVertex* nq_recv_buf_; // shared memory (memory space is shared with work_buf_)

	int64_t* pred_; // passed from main method

	struct SharedDataSet {
		memory::SpinBarrier *sync;
		int *offset; // max(max_threads*2+1, 2*mpi.size_z+1)
	} s_; // shared memory

	int current_level_;
	bool forward_or_backward_;
	bool bitmap_or_list_;
	bool growing_or_shrinking_;
	bool packet_buffer_is_dirty_;

	struct {
		void* thread_local_;
		void* shared_memory_;
	} buffer_;
};

#undef debug

#endif /* BFS_HPP_ */

