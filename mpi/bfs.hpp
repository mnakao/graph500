/*
 * bfs.hpp
 *
 *  Created on: Mar 5, 2012
 *      Author: koji
 */

#ifndef BFS_HPP_
#define BFS_HPP_

#include <pthread.h>

#include <deque>

#if CUDA_ENABLED
#include "gpu_host.hpp"
#endif

#include "utils.hpp"
#include "double_linked_list.h"
#include "fiber.hpp"

#include "low_level_func.h"

namespace bfs_detail {
using namespace BFS_PARAMS;

enum VARINT_BFS_KIND {
	VARINT_FOLD,
	VARINT_EXPAND_CQ, // current queue
	VARINT_EXPAND_SV, // shared visited
};

struct PacketIndex {
	uint16_t length;
	uint16_t num_int;
};

struct CompressedStream {
	uint32_t packet_index_start;
	union {
		uint8_t stream[BLOCK_V0_LEGNTH];
		PacketIndex index[BLOCK_V0_LEGNTH/sizeof(PacketIndex)];
	} d;
};

struct FoldCommBuffer {
	FoldEdge* edges;
	// info
	uint8_t complete_flag; // flag is set when send or receive complete. 0: plist, 1: vlist
	int target; // rank to which send or from which receive
	int num_packets;
	int num_edges;
	int v0_stream_length;
	ListEntry free_link;
	ListEntry extra_buf_link;
};
/*
struct FoldPacket {
	int num_edges;
	int64_t v0_list[PACKET_LENGTH];
	uint32_t v1_list[PACKET_LENGTH]; // SCALE - lgsize <= 32
};
*/
inline int get_num_packet(CompressedStream* stream, int stream_length, bool pair_packet)
{
	return (stream_length - offsetof(CompressedStream, d)) / sizeof(PacketIndex)
			- stream->packet_index_start;
}

class Bfs2DComm
{
public:
	class EventHandler {
	public:
		virtual ~EventHandler() { }
		virtual void fold_received(FoldCommBuffer* data) = 0;
		virtual void fold_finish() = 0;
	};

	class Communicatable {
	public:
		virtual ~Communicatable() { }
		virtual void comm() = 0;
	};

	Bfs2DComm(EventHandler* event_handler, bool cuda_enabled, FiberManager* fiber_man__)
		: event_handler_(event_handler)
		, cuda_enabled_(cuda_enabled)
		, fiber_man_(fiber_man__)
	{
		d_ = new DynamicDataSet();
		pthread_mutex_init(&d_->thread_sync_, NULL);
		pthread_cond_init(&d_->thread_state_,  NULL);
		pthread_cond_init(&d_->send_buffer_cond_,  NULL);
		d_->cleanup_ = false;
		d_->command_active_ = false;
		d_->suspended_ = false;
		d_->terminated_ = false;
#if PROFILING_MODE
		d_->num_extra_buffer_ = 0;
#endif
		initializeListHead(&d_->free_buffer_);
		initializeListHead(&d_->extra_buffer_);

		// check whether the size of CompressedStream is page-aligned
		// which v1_list of FoldCommBuffer needs to page-align. allocate_fold_comm_buffer
		const int foldcomm_width = roundup<CACHE_LINE>(sizeof(FoldCommBuffer));
		const int foldbuf_width = sizeof(FoldEdge) * BLOCK_V1_LENGTH;

		comm_length_ =
#if BFS_BACKWARD
				std::max(mpi.size_2dc, mpi.size_2dr);
#else
				mpi.size_2dc;
#endif
#if VERVOSE_MODE
		if(mpi.isMaster()) {
			fprintf(IMD_OUT, "Allocating communication buffer (%zu * %d bytes)\n",
					(size_t)foldbuf_width*PRE_ALLOCATE_COMM_BUFFER*comm_length_, mpi.size_2d);
		}
#endif
		buffer_.fold_comm_ = cache_aligned_xcalloc(foldcomm_width*PRE_ALLOCATE_COMM_BUFFER*comm_length_);
		buffer_.fold_ = page_aligned_xcalloc((size_t)foldbuf_width*PRE_ALLOCATE_COMM_BUFFER*comm_length_);
		fold_node_ = (FoldNode*)malloc(sizeof(fold_node_[0])*comm_length_);
		fold_node_comm_ = (FoldNodeComm*)malloc(sizeof(fold_node_comm_[0])*comm_length_);
		mpi_reqs_ = (MPI_Request*)malloc(sizeof(mpi_reqs_[0])*comm_length_*REQ_TOTAL);

#if CUDA_ENABLED
		if(cuda_enabled_) {
			CudaStreamManager::begin_cuda();
			CUDA_CHECK(cudaHostRegister(buffer_.fold_, foldbuf_width*PRE_ALLOCATE_COMM_BUFFER*mpi.size_2dc, cudaHostRegisterPortable));
			CudaStreamManager::end_cuda();
		}
#endif

		d_->num_send_reserved_buffer_ = 0;
		d_->num_recv_reserved_buffer_ = 0;
		for(int i = 0; i < comm_length_; ++i) {
			pthread_mutex_init(&fold_node_[i].send_mutex, NULL);

			initializeListHead(&fold_node_[i].sending_buffer);

			FoldCommBuffer *buf[PRE_ALLOCATE_COMM_BUFFER];
			for(int k = 0; k < PRE_ALLOCATE_COMM_BUFFER; ++k) {
				buf[k] = (FoldCommBuffer*)((uint8_t*)buffer_.fold_comm_ + foldcomm_width*(i*PRE_ALLOCATE_COMM_BUFFER + k));
				buf[k]->edges = (FoldEdge*)((uint8_t*)buffer_.fold_ + foldbuf_width*(i*PRE_ALLOCATE_COMM_BUFFER + k));
				buf[k]->num_edges = 0;
				buf[k]->num_packets = 0;
				buf[k]->v0_stream_length = 0;
				initializeListEntry(&buf[k]->free_link);
				initializeListEntry(&buf[k]->extra_buf_link);
			}

			fold_node_[i].current = NULL;
			for(int k = 0; k < PRE_ALLOCATE_COMM_BUFFER; ++k) {
				listInsertBack(&d_->free_buffer_, &buf[k]->free_link);
			}

			for(int k = 0; k < REQ_TOTAL; ++k) {
				mpi_reqs_[REQ_TOTAL*i + k] = MPI_REQUEST_NULL;
			}
			fold_node_comm_[i].recv_buffer = NULL;
			fold_node_comm_[i].send_buffer = NULL;
		}

		// initial value
		d_->fold_comm_ = mpi.comm_2dr;
		d_->fold_comm_size_ = mpi.size_2dc;
	//	d_->fold_sending_count_ = mpi.size_2dc;

		pthread_create(&d_->thread_, NULL, comm_thread_routine_, this);
	}

	virtual ~Bfs2DComm()
	{
		if(!d_->cleanup_) {
			d_->cleanup_ = true;
			pthread_mutex_lock(&d_->thread_sync_);
			d_->terminated_ = true;
			d_->suspended_ = true;
			d_->command_active_ = true;
			pthread_mutex_unlock(&d_->thread_sync_);
			pthread_cond_broadcast(&d_->thread_state_);
			pthread_join(d_->thread_, NULL);
			pthread_mutex_destroy(&d_->thread_sync_);
			pthread_cond_destroy(&d_->thread_state_);
			pthread_cond_destroy(&d_->send_buffer_cond_);

			for(int i = 0; i < comm_length_; ++i) {
				pthread_mutex_destroy(&fold_node_[i].send_mutex);
			}

			release_extra_buffer();

#if CUDA_ENABLED
			if(cuda_enabled_) {
				CudaStreamManager::begin_cuda();
				CUDA_CHECK(cudaHostUnregister(buffer_.fold_));
				CudaStreamManager::end_cuda();
			}
#endif

			free(buffer_.fold_comm_); buffer_.fold_comm_ = NULL;
			free(buffer_.fold_); buffer_.fold_ = NULL;
			free(fold_node_); fold_node_ = NULL;
			free(fold_node_comm_); fold_node_comm_ = NULL;
			free(mpi_reqs_); mpi_reqs_ = NULL;

			delete d_; d_ = NULL;
		}
	}

	void release_extra_buffer() {
		while(listIsEmpty(&d_->extra_buffer_) == false) {
			FoldCommBuffer* sb = CONTAINING_RECORD(d_->extra_buffer_.fLink,
					FoldCommBuffer, extra_buf_link);
			listRemove(&sb->extra_buf_link);
#if CUDA_ENABLED
			if(cuda_enabled_) {
				CudaStreamManager::begin_cuda();
				CUDA_CHECK(cudaHostUnregister(sb));
				CudaStreamManager::end_cuda();
			}
#endif
			free(sb);
		}
#if PROFILING_MODE
		d_->num_extra_buffer_ = 0;
#endif
	}

	void begin_fold_comm(bool forward_or_backward)
	{
#if !BFS_BACKWARD
		assert (forward_or_backward);
#endif
		CommCommand cmd;
		cmd.kind = FOLD_SEND_START;
		d_->forward_or_backward_ = forward_or_backward;
		if(forward_or_backward) { // forward
			d_->fold_comm_ = mpi.comm_2dr;
			d_->fold_comm_size_ = mpi.size_2dc;
	//		d_->fold_sending_count_ = mpi.size_2dc;
		}
		else { // backward
			d_->fold_comm_ = mpi.comm_2dc;
			d_->fold_comm_size_ = mpi.size_2dr;
	//		d_->fold_sending_count_ = mpi.size_2dr;
		}
		put_command(cmd);
	}

	template <bool need_lock>
	FoldCommBuffer* lock_fold_buffer(int dest_c, int num_edges) {
		FoldNode& dest_node = fold_node_[dest_c];
		while(1) {
			if(need_lock) pthread_mutex_lock(&dest_node.send_mutex);
			if(dest_node.current == NULL) {
				dest_node.current = fold_get_buffer(true /* send buffer */);
				if(dest_node.current == NULL) {
					if(need_lock) pthread_mutex_unlock(&dest_node.send_mutex);
					while(fiber_man_->process_task(1)); // process recv task
					continue;
				}
			}
			FoldCommBuffer* sb = dest_node.current;
			if(sb->num_edges + num_edges > BLOCK_V1_LENGTH)  {
				assert (sb->num_edges <= BLOCK_V1_LENGTH);
				fold_send_submit(dest_c);
				sb = dest_node.current = fold_get_buffer(true /* send buffer */);
				if(sb == NULL) {
					if(need_lock) pthread_mutex_unlock(&dest_node.send_mutex);
					while(fiber_man_->process_task(1)); // process recv task
					continue;
				}
			}
			return sb;
		}
	}

	void fold_send(
			FoldEdge* edges,
			int num_edges,
			int dest_c)
	{
		assert(num_edges > 0);

		FoldCommBuffer* sb = lock_fold_buffer<true>(dest_c, num_edges);
		FoldNode& dest_node = fold_node_[dest_c];
		// add to send buffer
		memcpy(sb->edges + sb->num_edges, edges, sizeof(edges[0])*num_edges);
		sb->num_edges += num_edges;
		pthread_mutex_unlock(&dest_node.send_mutex);
	}

	void fold_send_end(int dest_c)
	{
#if PROFILING_MODE
		profiling::TimeKeeper tk_wait;
#endif
		FoldNode& dest_node = fold_node_[dest_c];
		FoldCommBuffer* sb = lock_fold_buffer<true>(dest_c, BLOCK_V1_LENGTH);
		assert(sb->num_edges == 0);
		fold_send_submit(dest_c);
		dest_node.current = NULL;

		pthread_mutex_unlock(&dest_node.send_mutex);
/*
		if(__sync_fetch_and_add(&d_->fold_sending_count_, -1) == 1) {
			CommCommand cmd;
			cmd.kind = FOLD_SEND_END;
			put_command(cmd);
		}
		*/
#if PROFILING_MODE
		comm_time_ += tk_wait;
#endif
	}

	void input_command(Communicatable* comm)
	{
		CommCommand cmd;
		cmd.kind = MANUAL_COMM;
		cmd.cmd = comm;
		put_command(cmd);
	}

	void relase_fold_buffer(FoldCommBuffer* buf)
	{
		fold_put_buffer(false /* receive buffer */, buf);
	}

	int64_t expand_max_cq_size(int64_t cq_size) const
	{
		int64_t max_cq_size;
	    MPI_Allreduce(&cq_size, &max_cq_size, 1, get_mpi_type(cq_size), MPI_MAX, mpi.comm_2dc);
	    return max_cq_size;
	}
#if 0
	template <typename BITMAP_TYPE>
	void expand_comm_bitmap(const BITMAP_TYPE* local, int64_t length, BITMAP_TYPE* global) const
	{
		MPI_Allgather(local, length, MpiTypeOf<BITMAP_TYPE>::type,
				global, length, MpiTypeOf<BITMAP_TYPE>::type, mpi.comm_2dc);
	}

	void expand_comm_stream(const uint8_t* local, int length, uint8_t* global, int* offset) const
	{
		int recv_count[mpi.size_2dc];
		MPI_Allgather(&length, 1, MPI_INT, recv_count, 1, MPI_INT, mpi.comm_2dc);
		offset[0] = 0;
		for(int i = 0; i < mpi.size_2dc; ++i) offset[i + 1] = offset[i] + recv_count[i];
		MPI_Allgatherv(const_cast<uint8_t*>(local), length, MPI_BYTE, global, recv_count, offset, MPI_BYTE, mpi.comm_2dc);
	}
#endif
#if PROFILING_MODE
	void submit_prof_info(const char* str, int number) {
		comm_time_.submit(str, number);
	}
	void submit_mem_info() {
		profiling::g_pis.submitCounter(d_->num_extra_buffer_, "# of extra buffer", 0);
	}
#endif
#ifndef NDEBUG
	bool check_num_send_buffer() { return (d_->num_send_reserved_buffer_ == 0 && d_->num_recv_reserved_buffer_ == 0); }
#endif
private:
	EventHandler* event_handler_;
	bool cuda_enabled_;

	enum COMM_COMMAND {
		FOLD_SEND_START,
		FOLD_SEND,
	//	FOLD_SEND_END,
		MANUAL_COMM,
	};

	struct CommCommand {
		COMM_COMMAND kind;
		union {
			// FOLD_SEND
			int dest_c;
			// COMM_COMMAND
			Communicatable* cmd;
		};
	};

	struct DynamicDataSet {
		// lock topology
		// FoldNode::send_mutex -> thread_sync_
		pthread_t thread_;
		pthread_mutex_t thread_sync_;
		pthread_cond_t thread_state_;
		pthread_cond_t send_buffer_cond_;

		bool cleanup_;

		// monitor : thread_sync_
		volatile bool command_active_;
		volatile bool suspended_;
		volatile bool terminated_;
		std::deque<CommCommand> command_queue_;
		ListEntry free_buffer_;

		// accessed by comm thread only
		ListEntry extra_buffer_;
		int num_send_reserved_buffer_;
		int num_recv_reserved_buffer_;
#if PROFILING_MODE
		int num_extra_buffer_;
#endif

		bool forward_or_backward_;
		MPI_Comm fold_comm_;
		int fold_comm_size_;
	//	int fold_sending_count_;
	} *d_;

	struct FoldNode {
		pthread_mutex_t send_mutex;

		// monitor : send_mutex
#if VLQ_COMPRESSION
		PacketIndex sb_index[MAX_PACKETS_PER_BLOCK];
#endif
		FoldCommBuffer* current;

		// monitor : thread_sync_
		ListEntry sending_buffer;
	};

	struct FoldNodeComm {
		FoldCommBuffer* recv_buffer;
		FoldCommBuffer* send_buffer;
	};

	enum MPI_REQ_INDEX {
		REQ_SEND = 0,
		REQ_RECV = 1,
		REQ_TOTAL = 2,
	};

	int comm_length_;
	FoldNode* fold_node_;
	FoldNodeComm* fold_node_comm_;
	FiberManager* fiber_man_;
	std::deque<int> recv_stv;

	// accessed by communication thread only
	MPI_Request* mpi_reqs_;

	struct {
		void* fold_comm_;
		void* fold_;
	} buffer_;
#if PROFILING_MODE
	profiling::TimeSpan comm_time_;
#endif

	static void* comm_thread_routine_(void* pThis) {
		static_cast<Bfs2DComm*>(pThis)->comm_thread_routine();
		pthread_exit(NULL);
	}
	void comm_thread_routine()
	{
		int num_recv_active = 0, num_send_active = 0;

		// command loop
		while(true) {
			if(d_->command_active_) {
				pthread_mutex_lock(&d_->thread_sync_);
				CommCommand cmd;
				while(pop_command(&cmd)) {
					pthread_mutex_unlock(&d_->thread_sync_);
					switch(cmd.kind) {
					case FOLD_SEND_START:
						for(int i = 0; i < d_->fold_comm_size_; ++i) {
							FoldCommBuffer* rb = fold_get_buffer(false /* receive buffer */);
							assert (rb != NULL);
							fold_set_receive_buffer(i, rb);
						}
						num_send_active = num_recv_active = d_->fold_comm_size_;
						break;
					case FOLD_SEND:
						fold_set_send_buffer(cmd.dest_c);
						break;
						/*
					case FOLD_SEND_END:
						if(num_recv_active == 0 && num_send_active == 0) {
							assert (d_->num_send_reserved_buffer_ == comm_length_);
							event_handler_->fold_finish();
						}
						break;
						*/
					case MANUAL_COMM:
						cmd.cmd->comm();
						break;
					}
					pthread_mutex_lock(&d_->thread_sync_);
				}
				pthread_mutex_unlock(&d_->thread_sync_);
			}
			if(num_recv_active == 0 && num_send_active == 0) {
				pthread_mutex_lock(&d_->thread_sync_);
				if(d_->command_active_ == false) {
					d_->suspended_ = true;
					if( d_->terminated_ ) { pthread_mutex_unlock(&d_->thread_sync_); break; }
					pthread_cond_wait(&d_->thread_state_, &d_->thread_sync_);
				}
				pthread_mutex_unlock(&d_->thread_sync_);
			}

			int index;
			int flag;
			MPI_Status status;
			MPI_Testany(d_->fold_comm_size_ * (int)REQ_TOTAL, mpi_reqs_, &index, &flag, &status);

			if(flag == 0 || index == MPI_UNDEFINED) {
				continue;
			}

			const int src_c = index/REQ_TOTAL;
			const MPI_REQ_INDEX req_kind = (MPI_REQ_INDEX)(index%REQ_TOTAL);
			const bool b_send = (req_kind == REQ_SEND);

			FoldNodeComm& comm_node = fold_node_comm_[src_c];
			FoldCommBuffer* buf = b_send ? comm_node.send_buffer : comm_node.recv_buffer;

			assert (mpi_reqs_[index] == MPI_REQUEST_NULL);
			mpi_reqs_[index] = MPI_REQUEST_NULL;

			switch(req_kind) {
			case REQ_RECV:
				{
					MPI_Get_count(&status, MpiTypeOf<uint32_t>::type, &buf->num_edges);
#if VERVOSE_MODE
					g_fold_recv += buf->num_edges * sizeof(uint32_t);
#endif
					buf->num_edges /= 3;
				}
				break;
			default:
				break;
			}

			bool completion_message = (buf->num_edges == 0);
			// complete
			if(b_send) {
				// send buffer
				comm_node.send_buffer = NULL;
				fold_put_buffer(true /* send buffer */, buf);
				if(completion_message) {
					// sent fold completion
					--num_send_active;
					if(num_recv_active == 0 && num_send_active == 0) {
						event_handler_->fold_finish();
					}
				}
				else {
					fold_set_send_buffer(src_c);
				}
			}
			else {
				// recv buffer
				if(completion_message) {
					// received fold completion
					--num_recv_active;
					fold_node_comm_[src_c].recv_buffer = NULL;
					fold_put_buffer(false /* receive buffer */, buf);
					if(num_recv_active == 0 && num_send_active == 0) {
						event_handler_->fold_finish();
					}
				}
				else {
					// received both plist and vlist
					// set new buffer for next receiving
					recv_stv.push_back(src_c);

					event_handler_->fold_received(buf);
				}
			}

			// process recv starves
			while(recv_stv.size() > 0) {
				FoldCommBuffer *rb = fold_get_buffer(false /* receive buffer */);
				if(rb == NULL) break;
				fold_set_receive_buffer(recv_stv.front(), rb);
				recv_stv.pop_front();
			}

		}
	}

	bool pop_command(CommCommand* cmd) {
		if(d_->command_queue_.size()) {
			*cmd = d_->command_queue_[0];
			d_->command_queue_.pop_front();
			return true;
		}
		d_->command_active_ = false;
		return false;
	}

	void put_command(CommCommand& cmd)
	{
		bool command_active;

		pthread_mutex_lock(&d_->thread_sync_);
		d_->command_queue_.push_back(cmd);
		command_active = d_->command_active_;
		if(command_active == false) d_->command_active_ = true;
		pthread_mutex_unlock(&d_->thread_sync_);

		if(command_active == false) pthread_cond_broadcast(&d_->thread_state_);
	}

	FoldCommBuffer* allocate_fold_buffer()
	{
		const int edges_offset = roundup<PAGE_SIZE>(sizeof(FoldCommBuffer));
		const int mem_length = edges_offset + BLOCK_V1_LENGTH * sizeof(FoldEdge);
		uint8_t* new_buffer = (uint8_t*)page_aligned_xcalloc(mem_length);
#if CUDA_ENABLED
		if(cuda_enabled_) {
			CudaStreamManager::begin_cuda();
			CUDA_CHECK(cudaHostRegister(new_buffer, mem_length, cudaHostRegisterPortable));
			CudaStreamManager::end_cuda();
		}
#endif
		FoldCommBuffer* r = (FoldCommBuffer*)new_buffer;
		r->edges = (FoldEdge*)(new_buffer + edges_offset);
		initializeListEntry(&r->free_link);
		initializeListEntry(&r->extra_buf_link);
#if 0
		memset(r->v0_stream, 0, sizeof(CompressedStream));
		memset(r->v1_list, 0, BLOCK_V1_LENGTH*sizeof(r->v1_list[0]));
		memset(r, 0, sizeof(FoldCommBuffer));
		r->v0_stream = (CompressedStream*)(new_buffer + v0_offset);
		r->v1_list = (uint32_t*)(new_buffer + v1_offset);
#endif
		return r;
	}

	bool is_send_buffer_available() {
		// # of remaining sending buffer
		if(d_->num_send_reserved_buffer_ >= comm_length_ * MAX_EXTRA_SEND_BUFFER)
			return false;
		// # of recv tasks
		if(d_->num_recv_reserved_buffer_ - comm_length_ > 100)
			return false;
		// do not permit allocate new buffer for send
		if(listIsEmpty(&d_->free_buffer_))
			return false;
		return true;
	}

	FoldCommBuffer* fold_get_buffer(bool send_or_recv)
	{
		pthread_mutex_lock(&d_->thread_sync_);
		if(send_or_recv) {
			if(!is_send_buffer_available()) {
				pthread_mutex_unlock(&d_->thread_sync_);
				return NULL;
			}
			++(d_->num_send_reserved_buffer_);
		}
		else {
			++(d_->num_recv_reserved_buffer_);
		}
#if 1
		while(listIsEmpty(&d_->free_buffer_)) {
			pthread_mutex_unlock(&d_->thread_sync_);
			fiber_man_->process_task(1);
			pthread_mutex_lock(&d_->thread_sync_);
		}
#else
		if(listIsEmpty(&d_->free_buffer_)) {
			pthread_mutex_unlock(&d_->thread_sync_);
			FoldCommBuffer* new_buffer = allocate_fold_buffer();
			pthread_mutex_lock(&d_->thread_sync_);
			listInsertBack(&d_->extra_buffer_, &new_buffer->extra_buf_link);
#if PROFILING_MODE
			d_->num_extra_buffer_++;
#endif
			pthread_mutex_unlock(&d_->thread_sync_);
			return new_buffer;
		}
#endif
		FoldCommBuffer* buffer = CONTAINING_RECORD(d_->free_buffer_.fLink,
				FoldCommBuffer, free_link);
		listRemove(&buffer->free_link);
		pthread_mutex_unlock(&d_->thread_sync_);
		return buffer;
	}

	void fold_put_buffer(bool send_or_recv, FoldCommBuffer* buf)
	{
		bool signal_send_wait = false;
		buf->num_edges = buf->num_packets = buf->v0_stream_length = 0;
		pthread_mutex_lock(&d_->thread_sync_);
		listInsertBack(&d_->free_buffer_, &buf->free_link);
		if(send_or_recv) {
			if(d_->num_send_reserved_buffer_ >= comm_length_ * MAX_EXTRA_SEND_BUFFER) {
				signal_send_wait = true;
			}
			--(d_->num_send_reserved_buffer_);
		}
		else {
			--(d_->num_recv_reserved_buffer_);
		}
		pthread_mutex_unlock(&d_->thread_sync_);
	//	if(signal_send_wait) pthread_cond_broadcast(&d_->send_buffer_cond_);
	}

	void fold_set_receive_buffer(int src_c, FoldCommBuffer *rb)
	{
		FoldNodeComm& comm_node = fold_node_comm_[src_c];
		MPI_Request* recv_reqs = &mpi_reqs_[REQ_TOTAL*src_c + REQ_RECV];

		comm_node.recv_buffer = rb;
		MPI_Irecv(rb->edges, BLOCK_V1_LENGTH*3,
				MpiTypeOf<uint32_t>::type, src_c, COMM_V0_TAG, d_->fold_comm_, recv_reqs);

		rb->complete_flag = 0;
	}

	void fold_set_send_buffer(int dest_c)
	{
		FoldNode& node = fold_node_[dest_c];
		FoldNodeComm& comm_node = fold_node_comm_[dest_c];
		FoldCommBuffer* sb = NULL;

		if(comm_node.send_buffer) {
			return ;
		}

		pthread_mutex_lock(&d_->thread_sync_);
		if(listIsEmpty(&node.sending_buffer) == false) {
			sb = CONTAINING_RECORD(node.sending_buffer.fLink, FoldCommBuffer, free_link);
			listRemove(&sb->free_link);
		}
		pthread_mutex_unlock(&d_->thread_sync_);

		if(sb) {
			assert (sb->num_edges <= BLOCK_V1_LENGTH);
			comm_node.send_buffer = sb;
			MPI_Isend(sb->edges, sb->num_edges*3,
					MpiTypeOf<uint32_t>::type, dest_c, COMM_V0_TAG, d_->fold_comm_, &mpi_reqs_[REQ_TOTAL*dest_c + REQ_SEND]);

			sb->complete_flag = 0;
		}

	}

	void fold_send_submit(int dest_c)
	{
		FoldNode& dest_node = fold_node_[dest_c];
		FoldCommBuffer* sb = dest_node.current;
	//	FoldCommBuffer* sb = CONTAINING_RECORD(dest_node.free_buffer.fLink, FoldCommBuffer, free_link);
#if VLQ_COMPRESSION
		int packet_index_start = get_blocks<sizeof(PacketIndex)>(sb->v0_stream_length);
		sb->v0_stream->packet_index_start = packet_index_start;
		// copy sb_index to tail plist
		memcpy(&sb->v0_stream->d.index[packet_index_start],
				dest_node.sb_index, sizeof(dest_node.sb_index[0])*sb->num_packets);
#endif
		sb->target = dest_c;
	#if 0
		ffprintf(IMD_OUT, stderr, "send[from=%d,to=%d,npacket=%d,ptr_packet_index=%d,vlist_length=%d]\n",
				rank, sb->target, sb->npacket, sb->ptr_packet_index, sb->vlist_length);
	#endif

	//	listRemove(&sb->free_link);

		CommCommand cmd;
		cmd.kind = FOLD_SEND;
		cmd.dest_c = dest_c;
		bool command_active;

		pthread_mutex_lock(&d_->thread_sync_);
		listInsertBack(&dest_node.sending_buffer, &sb->free_link);
		d_->command_queue_.push_back(cmd);
		command_active = d_->command_active_;
		if(command_active == false) d_->command_active_ = true;
		pthread_mutex_unlock(&d_->thread_sync_);

		if(command_active == false) pthread_cond_broadcast(&d_->thread_state_);
	}
};

} // namespace detail {


template <typename IndexArray, typename LocalVertsIndex, typename PARAMS>
class BfsBase
	: private bfs_detail::Bfs2DComm::EventHandler
{
public:
	typedef typename PARAMS::BitmapType BitmapType;
	typedef BfsBase<IndexArray, LocalVertsIndex, PARAMS> ThisType;
	enum {
		// parameters
		// Number of edge lists packed as 1 edge list.
		// Since manipulating 64 bit integer value in NVIDIA GPU is not efficient,
		// max size of this parameter is limited to 32.
		LOG_PACKING_EDGE_LISTS = PARAMS::LOG_PACKING_EDGE_LISTS, // 2^6 = 64
		// Number of CQ bitmap entries represent as 1 bit in summary.
		// Since the type of bitmap entry is int32_t and 1 cache line is composed of 32 bitmap entries,
		// 32 is effective value.
		LOG_CQ_SUMMARIZING = PARAMS::LOG_CQ_SUMMARIZING, // 2^4 = 16 -> sizeof(int64_t)*32 = 128bytes
		ENABLE_WRITING_DEPTH = 1,

		BATCH_PACKETS = 16,

		EXPAND_COMM_THRESOLD_AVG = 10, // 50%
		EXPAND_COMM_THRESOLD_MAX = 75, // 75%
		EXPAND_COMM_DENOMINATOR = 100,

		// non-parameters
		NUMBER_PACKING_EDGE_LISTS = (1 << LOG_PACKING_EDGE_LISTS),
		NUMBER_CQ_SUMMARIZING = (1 << LOG_CQ_SUMMARIZING),

		// "sizeof(BitmapType) * 8" is the number of bits in 1 entry of summary.
		MINIMUN_SIZE_OF_CQ_BITMAP = NUMBER_CQ_SUMMARIZING * sizeof(BitmapType) * 8,
	};

	BfsBase(bool cuda_enabled)
		: comm_(this, cuda_enabled, &fiber_man_)
#if 0
		, recv_task_(65536)
#endif
		, cq_comm_(this, true)
		, visited_comm_(this, false)
	{
		//
	}

	virtual ~BfsBase()
	{
		//
	}

	template <typename EdgeList>
	void construct(EdgeList* edge_list)
	{
		// minimun requirement of CQ
		// CPU: MINIMUN_SIZE_OF_CQ_BITMAP words -> MINIMUN_SIZE_OF_CQ_BITMAP * NUMBER_PACKING_EDGE_LISTS * mpi.size_2dc
		// GPU: THREADS_PER_BLOCK words -> THREADS_PER_BLOCK * NUMBER_PACKING_EDGE_LISTS * mpi.size_2dc

		if(mpi.isMaster()) {
			fprintf(IMD_OUT, "Index Type: %d bit per edge\n", IndexArray::bytes_per_edge * 8);
		}

		int log_min_vertices = get_msb_index(MINIMUN_SIZE_OF_CQ_BITMAP * NUMBER_PACKING_EDGE_LISTS * mpi.size_2d);
		graph_.log_packing_edge_lists_ = get_msb_index(NUMBER_PACKING_EDGE_LISTS);

		detail::GraphConstructor2DCSR<IndexArray, LocalVertsIndex, EdgeList> constructor;
#if VERTEX_SORTING
		constructor.construct(edge_list, log_min_vertices, true /* sorting vertex */, false, graph_);
#else
		constructor.construct(edge_list, log_min_vertices, false /* sorting vertex */, false, graph_);
#endif

		log_local_bitmap_ = std::max(0,
				std::max(graph_.log_local_verts() - LOG_PACKING_EDGE_LISTS,
					get_msb_index(MINIMUN_SIZE_OF_CQ_BITMAP) - get_msb_index(mpi.size_2dr)));
	}

	void prepare_bfs() {
		printInformation();
		allocate_memory(graph_);
	}

	void run_bfs(int64_t root, int64_t* pred);

	void get_pred(int64_t* pred) {
	//	comm_.release_extra_buffer();
	}

	void end_bfs() {
		deallocate_memory();
	}

	Graph2DCSR<IndexArray, LocalVertsIndex> graph_;

// protected:
	struct ThreadLocalBuffer {
		uint8_t stream_buffer[BFS_PARAMS::PACKET_LENGTH*7*BATCH_PACKETS]; // 48bit per vertex
		int64_t decode_buffer[BFS_PARAMS::PACKET_LENGTH*(BATCH_PACKETS+1)];
		bfs_detail::PacketIndex packet_index[BATCH_PACKETS];
		RowPairData row_buffer[BFS_PARAMS::BNL_ARRAY_LENGTH];
		FoldPacket fold_packet[1];
	};

	int64_t get_bitmap_size_v0() const {
		return (int64_t(1) << log_local_bitmap_) * mpi.size_2dr;
	}
	int64_t get_summary_size_v0() const {
		return get_bitmap_size_v0() / MINIMUN_SIZE_OF_CQ_BITMAP;
	}
	int64_t get_bitmap_size_v1() const {
		return (int64_t(1) << log_local_bitmap_) * mpi.size_2dc;
	}
	int64_t get_bitmap_size_visited() const {
		return (int64_t(1) << log_local_bitmap_);
	}
	int64_t get_number_of_local_vertices() const {
		return (int64_t(1) << graph_.log_local_verts());
	}
	int64_t get_actual_number_of_local_vertices() const {
		return (int64_t(1) << graph_.log_actual_local_verts());
	}
	int64_t get_number_of_global_vertices() const {
		return (int64_t(1) << graph_.log_global_verts());
	}
	int64_t get_actual_number_of_global_vertices() const {
		return (int64_t(1) << graph_.log_actual_global_verts());
	}

	// virtual functions
	virtual int varint_encode(const int64_t* input, int length, uint8_t* output, bfs_detail::VARINT_BFS_KIND kind)
	{
		return varint_encode_stream_signed((const uint64_t*)input, length, output);
	}

	int64_t get_nq_threshold()
	{
		return sizeof(BitmapType) * graph_.get_bitmap_size_visited() -
				sizeof(bfs_detail::PacketIndex) * (omp_get_max_threads()*2 + 16);
	}

	void compute_jobs_per_node(bool long_or_short, int num_dest_nodes, int& jobs_per_node, int64_t& chunk_size) {
		// long: 32KB chunk of CQ
		// short: 512KB chunk of CQ
		const int demon = long_or_short ? 8 : (8*16);
		const int maximum = get_blocks((long_or_short ? 2048 : 128), num_dest_nodes);
		const int summary_size_per_node = get_bitmap_size_visited() / MINIMUN_SIZE_OF_CQ_BITMAP;
		// compute raw value
		int raw_value = std::max(1, summary_size_per_node / demon);
		// apply upper limit
		jobs_per_node = std::min<int>(raw_value, maximum);
		chunk_size = summary_size_per_node / jobs_per_node;
	}

	template <typename JOB_TYPE>
	void internal_init_job(JOB_TYPE* job, int64_t offset, int64_t chunk, int64_t summary_size) {
		const int64_t start = std::min(offset * chunk, summary_size);
		const int64_t end = std::min(start + chunk, summary_size);
		job->this_ = this;
		job->i_start_ = start;
		job->i_end_ = end;
	}

	void allocate_memory(Graph2DCSR<IndexArray, LocalVertsIndex>& g)
	{
		const int max_threads = omp_get_max_threads();
	//	cq_bitmap_ = (BitmapType*)
	//			page_aligned_xcalloc(sizeof(cq_bitmap_[0])*get_bitmap_size_v0());
		cq_summary_ = (BitmapType*)
				malloc(sizeof(cq_summary_[0])*get_summary_size_v0());
	//	shared_visited_ = (BitmapType*)
	//			page_aligned_xcalloc(sizeof(shared_visited_[0])*get_bitmap_size_v1());
		nq_bitmap_ = (BitmapType*)
				page_aligned_xcalloc(sizeof(nq_bitmap_[0])*get_bitmap_size_visited());
#if VERTEX_SORTING
		nq_sorted_bitmap_ = (BitmapType*)
				page_aligned_xcalloc(sizeof(nq_bitmap_[0])*get_bitmap_size_visited());
#endif
		visited_ = (BitmapType*)
				page_aligned_xcalloc(sizeof(visited_[0])*get_bitmap_size_visited());

		tmp_packet_max_length_ = sizeof(BitmapType) *
				get_bitmap_size_visited() / BFS_PARAMS::PACKET_LENGTH + omp_get_max_threads()*2;
		tmp_packet_index_ = (bfs_detail::PacketIndex*)
				malloc(sizeof(bfs_detail::PacketIndex)*tmp_packet_max_length_);
#if BFS_EXPAND_COMPRESSION
		cq_comm_.local_buffer_ = (bfs_detail::CompressedStream*)
#else // #if BFS_EXPAND_COMPRESSION
		cq_comm_.local_buffer_ = (uint32_t*)
#endif // #if BFS_EXPAND_COMPRESSION
				page_aligned_xcalloc(sizeof(BitmapType)*get_bitmap_size_visited());
#if VERTEX_SORTING
#if BFS_EXPAND_COMPRESSION
		visited_comm_.local_buffer_ = (bfs_detail::CompressedStream*)
#else // #if BFS_EXPAND_COMPRESSION
		visited_comm_.local_buffer_ = (uint32_t*)
#endif // #if BFS_EXPAND_COMPRESSION
				page_aligned_xcalloc(sizeof(BitmapType)*get_bitmap_size_visited());
#else // #if VERTEX_SORTING
		visited_comm_.local_buffer_ = cq_comm_.local_buffer_;
#endif // #if VERTEX_SORTING
		cq_comm_.recv_buffer_ =
				page_aligned_xcalloc(sizeof(BitmapType)*get_bitmap_size_v0());
		visited_comm_.recv_buffer_ =
				page_aligned_xcalloc(sizeof(BitmapType)*get_bitmap_size_v1());

		thread_local_buffer_ = (ThreadLocalBuffer**)
				malloc(sizeof(thread_local_buffer_[0])*max_threads);
		d_ = (DynamicDataSet*)malloc(sizeof(d_[0]));
#if AVOID_BUSY_WAIT
		pthread_mutex_init(&d_->avoid_busy_wait_sync_, NULL);
#endif

		comm_length_ =
#if BFS_BACKWARD
				std::max(mpi.size_2dc, mpi.size_2dr);
#else
				mpi.size_2dc;
#endif
		const int buffer_width = roundup<CACHE_LINE>(
				sizeof(ThreadLocalBuffer) + sizeof(FoldPacket) * comm_length_);
		buffer_.thread_local_ = cache_aligned_xcalloc(buffer_width*max_threads);
		for(int i = 0; i < max_threads; ++i) {
			thread_local_buffer_[i] = (ThreadLocalBuffer*)
					((uint8_t*)buffer_.thread_local_ + buffer_width*i);
		}

		// compute job length
		const int num_nodes = mpi.size_2dr;
		const int node_idx = mpi.rank_2dr;
		int long_job_blocks, short_job_blocks;
		int64_t long_job_chunk, short_job_chunk;
		compute_jobs_per_node(true, num_nodes, long_job_blocks, long_job_chunk);
		sched_.long_job_length = long_job_blocks * num_nodes;
		compute_jobs_per_node(false, num_nodes, short_job_blocks, short_job_chunk);
		sched_.short_job_length = short_job_blocks * num_nodes;

		assert (long_job_blocks*long_job_chunk*num_nodes >= get_summary_size_v0());
		assert (short_job_blocks*short_job_chunk*num_nodes >= get_summary_size_v0());

		sched_.long_job = new ExtractEdge[sched_.long_job_length];
		sched_.short_job = new ExtractEdge[sched_.short_job_length];
#if BFS_BACKWARD
		sched_.back_long_job = new BackwardExtractEdge[sched_.long_job_length];
		sched_.back_short_job = new BackwardExtractEdge[sched_.short_job_length];
#endif
		const int64_t summary_size_v0 = get_summary_size_v0();
		assert ((sched_.long_job_length % long_job_blocks) == 0);
		assert ((sched_.long_job_length % long_job_blocks) == 0);
		for(int i = 0; i < num_nodes; ++i) {
#if 1
			int scrambled_node_idx = (i + node_idx) % num_nodes;
#else
			int scrambled_node_idx = i;
#endif
			for(int j = 0; j < long_job_blocks; ++j) {
				int i_node_major = i * long_job_blocks + j; // for forward jobs
				int i_block_major = j * num_nodes + scrambled_node_idx; // for backward jobs
				internal_init_job(&sched_.long_job[i_node_major], i_node_major, long_job_chunk, summary_size_v0);
#if BFS_BACKWARD
				internal_init_job(&sched_.back_long_job[i_block_major], i_node_major, long_job_chunk, summary_size_v0);
#endif
			}
			for(int j = 0; j < short_job_blocks; ++j) {
				int i_node_major = i * short_job_blocks + j; // for forward jobs
				int i_block_major = j * num_nodes + scrambled_node_idx; // for backward jobs
				internal_init_job(&sched_.short_job[i_node_major], i_node_major, short_job_chunk, summary_size_v0);
#if BFS_BACKWARD
				internal_init_job(&sched_.back_short_job[i_block_major], i_node_major, short_job_chunk, summary_size_v0);
#endif
			}
		}

		sched_.fold_end_job = new ExtractEnd[comm_length_];
		for(int i = 0; i < comm_length_; ++i) {
			sched_.fold_end_job[i].this_ = this;
			sched_.fold_end_job[i].dest_c_ = i;
		}

#if 0
		num_recv_tasks_ = max_threads * 3;
		for(int i = 0; i < num_recv_tasks_; ++i) {
			recv_task_.push(new ReceiverProcessing(this));
		}
#endif
#if BIT_SCAN_TABLE
		bit_scan_table_[0] = BFS_PARAMS::BIT_SCAN_TABLE_BITS;
		for(int i = 1; i < BFS_PARAMS::BIT_SCAN_TABLE_SIZE; ++i) {
			for(int b = 0; b < BFS_PARAMS::BIT_SCAN_TABLE_BITS; ++b) {
				if(i & (1 << b)) {
					bit_scan_table_[i] = b + 1;
					break;
				}
			}
		}
#endif
	}

	void deallocate_memory()
	{
		free(cq_bitmap_); cq_bitmap_ = NULL;
		free(cq_summary_); cq_summary_ = NULL;
		free(shared_visited_); shared_visited_ = NULL;
		free(nq_bitmap_); nq_bitmap_ = NULL;
#if VERTEX_SORTING
		free(nq_sorted_bitmap_); nq_sorted_bitmap_ = NULL;
#endif
		free(visited_); visited_ = NULL;
		free(tmp_packet_index_); tmp_packet_index_ = NULL;
		free(cq_comm_.local_buffer_); cq_comm_.local_buffer_ = NULL;
#if VERTEX_SORTING
		free(visited_comm_.local_buffer_); visited_comm_.local_buffer_ = NULL;
#endif
		free(cq_comm_.recv_buffer_); cq_comm_.recv_buffer_ = NULL;
		free(visited_comm_.recv_buffer_); visited_comm_.recv_buffer_ = NULL;
		free(thread_local_buffer_); thread_local_buffer_ = NULL;
#if AVOID_BUSY_WAIT
		pthread_mutex_destroy(&d_->avoid_busy_wait_sync_);
#endif
		free(d_); d_ = NULL;
		delete [] sched_.long_job; sched_.long_job = NULL;
		delete [] sched_.short_job; sched_.short_job = NULL;
#if BFS_BACKWARD
		delete [] sched_.back_long_job; sched_.back_long_job = NULL;
		delete [] sched_.back_short_job; sched_.back_short_job = NULL;
#endif
		delete [] sched_.fold_end_job; sched_.fold_end_job = NULL;
		free(buffer_.thread_local_); buffer_.thread_local_ = NULL;
#if 0
		for(int i = 0; i < num_recv_tasks_; ++i) {
			delete recv_task_.pop();
		}
#endif
	}

	void initialize_memory(int64_t* pred)
	{
		using namespace BFS_PARAMS;
		const int64_t num_local_vertices = get_actual_number_of_local_vertices();
		const int64_t bitmap_size_visited = get_bitmap_size_visited();
		const int64_t bitmap_size_v1 = get_bitmap_size_v1();
		const int64_t summary_size = get_summary_size_v0();

		BitmapType* cq_bitmap = cq_bitmap_;
		BitmapType* cq_summary = cq_summary_;
		BitmapType* nq_bitmap = nq_bitmap_;
#if VERTEX_SORTING
		BitmapType* nq_sorted_bitmap = nq_sorted_bitmap_;
#endif
		BitmapType* visited = visited_;
		BitmapType* shared_visited = shared_visited_;

#pragma omp parallel
		{
#if 1	// Only Spec2010 needs this initialization
#pragma omp for nowait
			for(int64_t i = 0; i < num_local_vertices; ++i) {
				pred[i] = -1;
			}
#endif
			// clear NQ and visited
#pragma omp for nowait
			for(int64_t i = 0; i < bitmap_size_visited; ++i) {
				nq_bitmap[i] = 0;
#if VERTEX_SORTING
				nq_sorted_bitmap[i] = 0;
#endif
				visited[i] = 0;
			}
			// clear CQ and CQ summary
#pragma omp for nowait
			for(int64_t i = 0; i < summary_size; ++i) {
				cq_summary[i] = 0;
				for(int k = 0; k < MINIMUN_SIZE_OF_CQ_BITMAP; ++k) {
					cq_bitmap[i*MINIMUN_SIZE_OF_CQ_BITMAP + k] = 0;
				}
			}
			// clear shared visited
#pragma omp for nowait
			for(int64_t i = 0; i < bitmap_size_v1; ++i) {
				shared_visited[i] = 0;
			}

			// clear fold packet buffer
			FoldPacket* packet_array = thread_local_buffer_[omp_get_thread_num()]->fold_packet;
			for(int i = 0; i < comm_length_; ++i) {
				packet_array[i].num_edges = 0;
			}
		}
	}

	int64_t get_compressed_stream_length(int64_t num_packets, int64_t content_length)
	{
#if DEBUG_PRINT
		if(mpi.isMaster()) printf("L:%d: get_compressed_stream_length: sizeof(bfs_detail::PacketIndex)=%d, offsetof(bfs_detail::CompressedStream, stream)=%d"
				", num_packets=%"PRId64", content_length=%"PRId64"\n", __LINE__,
				(int)sizeof(bfs_detail::PacketIndex), (int)offsetof(bfs_detail::CompressedStream, d), num_packets, content_length);
#endif
		return num_packets*sizeof(bfs_detail::PacketIndex) +
				roundup<sizeof(bfs_detail::PacketIndex)>(content_length) +
				offsetof(bfs_detail::CompressedStream, d);
	}

	bool expand_batch_wrtie_packet(int64_t* vertex_buffer, int num_vertices,
			int64_t threshold, bfs_detail::CompressedStream* output, bool cq_or_visited)
	{
		using namespace BFS_PARAMS;
		using namespace bfs_detail;

		ThreadLocalBuffer* tlb = thread_local_buffer_[omp_get_thread_num()];
		PacketIndex* local_packet_index = tlb->packet_index;
		uint8_t* local_stream_buffer = tlb->stream_buffer;
		// encode using varint
		int stream_buffer_offset = 0;
		int num_packets = get_blocks<PACKET_LENGTH>(num_vertices);
		for(int k = 0; k < num_packets; ++k) {
			int num_vertices_in_this_packet = std::min<int>(PACKET_LENGTH, num_vertices - k*PACKET_LENGTH);
			int64_t* vertex_buffer_k = vertex_buffer + k*PACKET_LENGTH;
			// compute differentials
#ifndef NDEBUG
			int64_t num_local_vertices = get_actual_number_of_local_vertices();
			for(int64_t r = 0; r < num_vertices_in_this_packet; ++r) {
				assert (vertex_buffer_k[r] >= 0);
				assert (vertex_buffer_k[r] < num_local_vertices);
			}
#endif
			for(int r = num_vertices_in_this_packet - 1; r > 0; --r) {
				vertex_buffer_k[r] -= vertex_buffer_k[r-1];
				assert (vertex_buffer_k[r] >= 0);
			}
			int length = varint_encode(vertex_buffer_k,
					num_vertices_in_this_packet, local_stream_buffer + stream_buffer_offset,
					cq_or_visited ? bfs_detail::VARINT_EXPAND_CQ : bfs_detail::VARINT_EXPAND_SV);
			local_packet_index[k].length = length;
			local_packet_index[k].num_int = num_vertices_in_this_packet;
#if DEBUG_PRINT
			if(mpi.isMaster()) printf("L:%d:SPacket (length,num_vertices)=(%d,%d)\n", __LINE__,
					length, num_vertices_in_this_packet);
#endif
			stream_buffer_offset += length;
		}
		// store to global buffer
		int64_t copy_packet_offset;
		int64_t copy_offset;
		int64_t total_num_packet;
		int64_t total_stream_length;
#pragma omp critical
		{
			copy_packet_offset = d_->num_tmp_packets_;
			copy_offset = d_->tmp_packet_offset_;
			total_num_packet = copy_packet_offset + num_packets;
			total_stream_length = copy_offset + stream_buffer_offset;
			d_->num_tmp_packets_ = total_num_packet;
			d_->tmp_packet_offset_ = total_stream_length;
		}
		if(get_compressed_stream_length
				(total_num_packet, total_stream_length) >= threshold)
		{
			return false;
		}
		memcpy(tmp_packet_index_ + copy_packet_offset,
				local_packet_index, sizeof(PacketIndex)*num_packets);
		memcpy(output->d.stream + copy_offset,
				local_stream_buffer, sizeof(local_stream_buffer[0]) * stream_buffer_offset);
		return true;
	}

#if BFS_EXPAND_COMPRESSION
	// return : true if creating succeeded,
	// 			false if stream length exceeded the length of bitmap
	bool expand_create_stream(const BitmapType* bitmap, bfs_detail::CompressedStream* output, int64_t* data_size, bool cq_or_visited)
	{
		using namespace BFS_PARAMS;
		using namespace bfs_detail;

		d_->num_tmp_packets_ = 0;
		d_->tmp_packet_offset_ = 0;

		const int64_t bitmap_size = get_bitmap_size_visited();
		const int64_t threshold = bitmap_size * sizeof(BitmapType);
		bool b_break = false;

#pragma omp parallel reduction(|:b_break)
		{
			int64_t* local_buffer = thread_local_buffer_[omp_get_thread_num()]->decode_buffer;
			int local_buffer_offset = 0;

			int64_t chunk_size = (bitmap_size + omp_get_num_threads() - 1) / omp_get_num_threads();
			int64_t i_start = chunk_size * omp_get_thread_num();
			int64_t i_end = std::min(i_start + chunk_size, bitmap_size);

			for(int64_t i = i_start; i < i_end; ++i) {
				const BitmapType bitmap_i = bitmap[i];
				if(bitmap_i == 0) {
					continue;
				}
				// read bitmap and store to temporary buffer
				for(int bit_idx = 0; bit_idx < NUMBER_PACKING_EDGE_LISTS; ++bit_idx) {
					if(bitmap_i & (int64_t(1) << bit_idx)) {
						local_buffer[local_buffer_offset++] = i*NUMBER_PACKING_EDGE_LISTS + bit_idx;
					}
				}

				if(local_buffer_offset >= PACKET_LENGTH*BATCH_PACKETS)
				{
					if(expand_batch_wrtie_packet(local_buffer,
							PACKET_LENGTH*BATCH_PACKETS, threshold, output, cq_or_visited) == false)
					{
						b_break = true;
						break;
					}
					local_buffer_offset -= PACKET_LENGTH*BATCH_PACKETS;
					memcpy(local_buffer, local_buffer + PACKET_LENGTH*BATCH_PACKETS,
							sizeof(local_buffer[0]) * local_buffer_offset);
				}
			}
			// flush local buffer
			if(b_break == false) {
				if(expand_batch_wrtie_packet(local_buffer,
						local_buffer_offset, threshold, output, cq_or_visited) == false)
				{
					b_break = true;
				}
			}
		} // #pragma omp parallel reduction(|:b_break)

		if(b_break) {
			*data_size = threshold;
			return false;
		}

		// write packet index
		int packet_index_start = get_blocks<sizeof(PacketIndex)>(d_->tmp_packet_offset_);
		output->packet_index_start = packet_index_start;
		// copy sb_index to tail plist
		memcpy(&output->d.index[packet_index_start],
				tmp_packet_index_, sizeof(tmp_packet_index_[0])*d_->num_tmp_packets_);

		*data_size = get_compressed_stream_length(d_->num_tmp_packets_, d_->tmp_packet_offset_);
#if DEBUG_PRINT
			if(mpi.isMaster()) printf("L:%d:Expand Send Packet (num_packet,stream_offset,packet_index_start)=(%d,%d,%d)\n", __LINE__,
					(int)d_->num_tmp_packets_, (int)d_->tmp_packet_offset_, packet_index_start);
#endif
		return true;
	}
#else // #if BFS_EXPAND_COMPRESSION
	// return : true if creating succeeded,
	// 			false if stream length exceeded the length of bitmap
	bool expand_create_stream(const BitmapType* bitmap, uint32_t* output, int64_t* data_size, bool cq_or_visited)
	{
		using namespace BFS_PARAMS;

		d_->tmp_packet_offset_ = 0; // use tmp_packet_offset_ as offset of output buffer

		const int64_t bitmap_size = get_bitmap_size_visited();

#pragma omp parallel
		{
			uint32_t* local_buffer = (uint32_t*)thread_local_buffer_[omp_get_thread_num()]->decode_buffer;
			int local_buffer_offset = 0;

			int64_t chunk_size = (bitmap_size + omp_get_num_threads() - 1) / omp_get_num_threads();
			int64_t i_start = chunk_size * omp_get_thread_num();
			int64_t i_end = std::min(i_start + chunk_size, bitmap_size);

			for(int64_t i = i_start; i < i_end; ++i) {
				const BitmapType bitmap_i = bitmap[i];
				if(bitmap_i == 0) {
					continue;
				}
				// read bitmap and store to temporary buffer
				for(int bit_idx = 0; bit_idx < NUMBER_PACKING_EDGE_LISTS; ++bit_idx) {
					if(bitmap_i & (BitmapType(1) << bit_idx)) {
						local_buffer[local_buffer_offset++] = i*NUMBER_PACKING_EDGE_LISTS + bit_idx;
					}
				}

				if(local_buffer_offset >= PACKET_LENGTH*BATCH_PACKETS)
				{
					int64_t offset = __sync_fetch_and_add(&d_->tmp_packet_offset_, local_buffer_offset);
					memcpy(output + offset, local_buffer, sizeof(output[0])*local_buffer_offset);
					local_buffer_offset = 0;
				}
			}
			// flush local buffer
			int64_t offset = __sync_fetch_and_add(&d_->tmp_packet_offset_, local_buffer_offset);
			memcpy(output + offset, local_buffer, sizeof(output[0])*local_buffer_offset);
		} // #pragma omp parallel

		*data_size = d_->tmp_packet_offset_;
#if DEBUG_PRINT
			if(mpi.isMaster()) printf("L:%d:Expand Send (num_vertices)=(%d)\n", __LINE__,
					(int)d_->tmp_packet_offset_);
#endif
		return true;
	}
#endif // #if BFS_EXPAND_COMPRESSION

	enum EXPAND_PHASE {
		EXPAND_FORWARD,
		EXPAND_SWITCH,
		EXPAND_BACKWARD,
	};

	struct ExpandCommCommand
		: public Runnable
		, public bfs_detail::Bfs2DComm::Communicatable
	{
		ExpandCommCommand(ThisType* this__, bool cq_or_visited)
			: this_(this__)
			, cq_or_visited_(cq_or_visited)
		{
			if(cq_or_visited) {
				// current queue
				comm_ = mpi.comm_2dc;
				comm_size_ = mpi.size_2dr;
				bitmap_size_in_bytes_ = this_->get_bitmap_size_v0() * sizeof(BitmapType);
			}
			else {
				// visited
				comm_ = mpi.comm_2dr;
				comm_size_ = mpi.size_2dc;
				bitmap_size_in_bytes_ = this_->get_bitmap_size_v1() * sizeof(BitmapType);
			}
			count_ = (int*)malloc(sizeof(int)*2*(comm_size_+1));
			offset_ = count_ + comm_size_+1;
			/*
			 * On changing to backward search,
			 * we change communication data, not communicator.
			 */
		}
		~ExpandCommCommand()
		{
			free(count_); count_ = NULL;
		}
		void start_comm(LocalVertsIndex local_data_size, bool force_bitmap, EXPAND_PHASE phase, bool exit_fiber_proc)
		{
			exit_fiber_proc_ = exit_fiber_proc;
			force_bitmap_ = force_bitmap;
			phase_ = phase;
			local_size_ = local_data_size;
			this_->comm_.input_command(this);
		}

#if BFS_EXPAND_COMPRESSION
		bfs_detail::CompressedStream* local_buffer_; // allocated by parent
#else
		uint32_t* local_buffer_;
#endif
		void* recv_buffer_; // allocated by parent

		virtual void comm() {

			gather_comm();

			this_->fiber_man_.submit(this, 0);
			if(exit_fiber_proc_) {
				this_->fiber_man_.end_processing();
			}
		}

		virtual void run() {
			const LocalVertsIndex local_bitmap_size = this_->get_bitmap_size_visited();

			if(cq_or_visited_) {
				// current queue
#if VERTEX_SORTING
				// clear NQ
				BitmapType* nq_bitmap = this_->nq_bitmap_;
#pragma omp parallel for
				for(int64_t i = 0; i < local_bitmap_size; ++i) {
					nq_bitmap[i] = 0;
				}
#endif
				BitmapType* summary = (phase_ == EXPAND_FORWARD) ? this_->cq_summary_ : NULL;
				if(stream_or_bitmap_) {
					// stream
					const LocalVertsIndex bitmap_size = this_->get_bitmap_size_v0();
					BitmapType* cq_bitmap = this_->cq_bitmap_;

					// update
					this_->d_->num_vertices_in_cq_ = this_->update_from_stream(
#if BFS_EXPAND_COMPRESSION
							static_cast<uint8_t*>(recv_buffer_),
#else // #if BFS_EXPAND_COMPRESSION
							static_cast<uint32_t*>(recv_buffer_),
#endif // #if BFS_EXPAND_COMPRESSION
							offset_, count_, comm_size_, cq_bitmap, bitmap_size, summary, cq_or_visited_);
				}
				else {
					// bitmap
					this_->d_->num_vertices_in_cq_ = this_->get_bitmap_size_v0() * NUMBER_PACKING_EDGE_LISTS;
					// fill 1 to summary
					if(summary) memset(summary, -1, sizeof(BitmapType) * this_->get_summary_size_v0());
				}
#ifndef NDEBUG
				if(phase_ != EXPAND_FORWARD) this_->check_shared_visited(phase_);
#endif // #ifndef NDEBUG
			}
			else {
				// shared visited
				// clear NQ
#if VERTEX_SORTING
				BitmapType* nq_bitmap = this_->nq_sorted_bitmap_;
#else
				BitmapType* nq_bitmap = this_->nq_bitmap_;
#endif
#pragma omp parallel for
				for(int64_t i = 0; i < local_bitmap_size; ++i) {
					nq_bitmap[i] = 0;
				}
#if !FAKE_VISITED_SHARING
				if(stream_or_bitmap_) {
					// stream
					this_->update_from_stream(
#if BFS_EXPAND_COMPRESSION
							static_cast<uint8_t*>(recv_buffer_),
#else // #if BFS_EXPAND_COMPRESSION
							static_cast<uint32_t*>(recv_buffer_),
#endif // #if BFS_EXPAND_COMPRESSION
							offset_, count_, comm_size_, this_->shared_visited_, this_->get_bitmap_size_v1(), NULL, cq_or_visited_);
				}
				else {
#if SHARED_VISITED_STRIPE
					// bitmap
					this_->update_shared_visited_from_bitmap(static_cast<BitmapType*>(recv_buffer_));
#endif
				}
#ifndef NDEBUG
				if(phase_ == EXPAND_FORWARD) this_->check_shared_visited(phase_);
#endif // #ifndef NDEBUG
#endif // #ifndef !FAKE_VISITED_SHARING
			}
		}

	protected:
		BitmapType* get_send_data() {
			switch(phase_) {
			case EXPAND_FORWARD:
				return cq_or_visited_ ? this_->nq_bitmap_ : this_->visited_;
			case EXPAND_SWITCH:
			case EXPAND_BACKWARD:
#if VERTEX_SORTING
				return cq_or_visited_ ? this_->visited_orig_bitmap_ : this_->nq_sorted_bitmap_;
#else // #if VERTEX_SORTING
				return cq_or_visited_ ? this_->visited_ : this_->nq_bitmap_;
#endif // #if VERTEX_SORTING
			}
			throw "!!!!Invalid Program!!!! : ExpandCommCommand::get_send_data()";
		}

#if BFS_EXPAND_COMPRESSION
		void gather_comm()
		{
			stream_or_bitmap_ = false;
			if(force_bitmap_ == false) {
				int max_size;
				int64_t bitmap_size_in_bytes = this_->get_bitmap_size_visited() * sizeof(BitmapType);
				int64_t threshold = int64_t((double)bitmap_size_in_bytes *
						(double)EXPAND_COMM_THRESOLD_MAX / (double)EXPAND_COMM_DENOMINATOR);
				MPI_Allreduce(&local_size_, &max_size, 1,
						get_mpi_type(local_size_), MPI_MAX, comm_);
				assert (get_mpi_type(local_size_) == get_mpi_type(max_size));
				if((int64_t)max_size <= threshold) {
					stream_or_bitmap_ = true;
				}
			}
			if(stream_or_bitmap_) {
				// transfer compressed stream
				MPI_Allgather(&local_size_, 1, get_mpi_type(local_size_),
						count_, 1, get_mpi_type(count_[0]), comm_);
				assert (get_mpi_type(local_size_) == get_mpi_type(count_[0]));
				offset_[0] = 0;
				for(int i = 0; i < comm_size_; ++i) {
					// take care of alignment !!!
					offset_[i+1] = roundup<sizeof(bfs_detail::PacketIndex)>
									(offset_[i] + count_[i]);
				}
#if DEBUG_PRINT
				if(mpi.isMaster()) printf("L:%d: Dump %d (len=%d)\n", __LINE__,
						((volatile uint16_t*)local_buffer_)[2], local_size_);
#endif
				MPI_Allgatherv(local_buffer_, local_size_, MPI_BYTE,
						recv_buffer_, count_, offset_, MPI_BYTE, comm_);
#if DEBUG_PRINT
				if(mpi.isMaster()) printf("L:%d: Dump %d\n", __LINE__,
						((volatile uint16_t*)recv_buffer_)[2]);
#endif
#if VERVOSE_MODE
				g_exs_send += local_size_;
				g_exs_recv += offset_[comm_size_];
#endif // #if VERVOSE_MODE
			}
			else {
				// transfer bitmap
				BitmapType* const bitmap = get_send_data();
				void* recv_buffer;
				if(cq_or_visited_) {
					// current queue
					recv_buffer = this_->cq_bitmap_;
				}
				else {
					// visited
#if FAKE_VISITED_SHARING
					return ;
#endif
#if SHARED_VISITED_STRIPE
					recv_buffer = recv_buffer_;
#else
					recv_buffer = this_->shared_visited_;
#endif
				}
				int bitmap_size = this_->get_bitmap_size_visited();
				// transfer bitmap
				MPI_Allgather(bitmap, bitmap_size, get_mpi_type(bitmap[0]),
						recv_buffer, bitmap_size, get_mpi_type(bitmap[0]), comm_);
#if VERVOSE_MODE
				g_bitmap_send += bitmap_size * sizeof(BitmapType);
				g_bitmap_recv += bitmap_size * comm_size_* sizeof(BitmapType);
#endif
			}
		}
#else // #if BFS_EXPAND_COMPRESSION
		void gather_comm()
		{
			stream_or_bitmap_ = false;
			if(force_bitmap_ == false) {
				// gather the number of vertices that each processor has
				MPI_Allgather(&local_size_, 1, get_mpi_type(local_size_),
						count_, 1, get_mpi_type(count_[0]), comm_);
				offset_[0] = 0;
				for(int i = 0; i < comm_size_; ++i) {
					offset_[i+1] = (offset_[i] + count_[i]);
				}
				int64_t threshold = int64_t((double)bitmap_size_in_bytes_ / sizeof(local_buffer_[0]) *
						(double)EXPAND_COMM_THRESOLD_MAX / (double)EXPAND_COMM_DENOMINATOR);
				if((int64_t)offset_[comm_size_] <= threshold) {
					stream_or_bitmap_ = true;
				}
			}
			if(stream_or_bitmap_) {
				MPI_Allgatherv(local_buffer_, local_size_, get_mpi_type(local_buffer_[0]),
						recv_buffer_, count_, offset_, get_mpi_type(local_buffer_[0]), comm_);
#if VERVOSE_MODE
				g_exs_send += local_size_ * sizeof(local_buffer_[0]);
				g_exs_recv += offset_[comm_size_] * sizeof(local_buffer_[0]);
#endif // #if VERVOSE_MODE
			}
			else {
				// transfer bitmap
				BitmapType* const bitmap = get_send_data();
				void* recv_buffer = cq_or_visited_ ? this_->cq_bitmap_ : this_->shared_visited_;
				int bitmap_size = this_->get_bitmap_size_visited();
				// transfer bitmap
				MPI_Allgather(bitmap, bitmap_size, get_mpi_type(bitmap[0]),
						recv_buffer, bitmap_size, get_mpi_type(bitmap[0]), comm_);
#if VERVOSE_MODE
				g_bitmap_send += bitmap_size * sizeof(BitmapType);
				g_bitmap_recv += bitmap_size * comm_size_* sizeof(BitmapType);
#endif
			}
		}
#endif // #if BFS_EXPAND_COMPRESSION

		ThisType* const this_;
		const bool cq_or_visited_;
		MPI_Comm comm_;
		int comm_size_;
		int64_t bitmap_size_in_bytes_;
		bool exit_fiber_proc_;
		bool force_bitmap_;
		bool stream_or_bitmap_;
		EXPAND_PHASE phase_;
		int* count_;
		int* offset_;
		// This should be LocalVertsIndex but
		// current MPI limit communication buffer size to 'int'.
		int local_size_;
	};

	struct UpdatePacketIndex {
		LocalVertsIndex offset;
		int16_t length;
		int16_t src_num;
#ifndef NDEBUG
		int16_t num_vertices;
#endif
	};

	void check_shared_visited(EXPAND_PHASE phase)
	{
		int64_t num_local_vertices = get_actual_number_of_local_vertices();
#if SHARED_VISITED_STRIPE
		const LocalVertsIndex src_num_factor = LocalVertsIndex(mpi.rank_2dc)*NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS;
		const int log_size_c = get_msb_index(mpi.size_2dc);
		const LocalVertsIndex mask2 = LocalVertsIndex(NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS) - 1;
		const LocalVertsIndex mask1 = get_number_of_local_vertices() - 1 - mask2;
#else
		int64_t word_idx_base;
		const BitmapType* bitmap;
		bool resolve_vertex_mapping = false;
		switch(phase) {
		case EXPAND_FORWARD:
			word_idx_base = get_bitmap_size_visited() * mpi.rank_2dc;
			bitmap = shared_visited_;
#if VERTEX_SORTING
			resolve_vertex_mapping = true;
#endif
			break;
		case EXPAND_SWITCH:
		case EXPAND_BACKWARD:
			word_idx_base = get_bitmap_size_visited() * mpi.rank_2dr;
			bitmap = cq_bitmap_;
			break;
		}

#endif
		for(int64_t i = 0; i < num_local_vertices; ++i) {
			int64_t v = resolve_vertex_mapping ? graph_.vertex_mapping_[i] : i;
#if SHARED_VISITED_STRIPE
			LocalVertsIndex sv_idx = src_num_factor | ((v & mask1) << log_size_c) | (v & mask2);
			int64_t word_idx = sv_idx / NUMBER_PACKING_EDGE_LISTS;
#else
			int64_t word_idx = word_idx_base + v / NUMBER_PACKING_EDGE_LISTS;
#endif
			int bit_idx = v % NUMBER_PACKING_EDGE_LISTS;
			bool visited = (bitmap[word_idx] & (BitmapType(1) << bit_idx)) != 0;
			assert((visited && (pred_[i] != -1)) || (!visited && (pred_[i] == -1)));
		}
	}

#if BFS_EXPAND_COMPRESSION
	// return number of vertices received
	int64_t update_from_stream(uint8_t* byte_stream, int* offset, int* count,
			int num_src, BitmapType* target, LocalVertsIndex length, BitmapType* summary, bool cq_or_visited /* for debug */)
	{
		LocalVertsIndex packet_count[num_src], packet_offset[num_src+1];
		packet_offset[0] = 0;

#ifndef NDEBUG
		const int rank_in_2d = (cq_or_visited ? mpi.rank_2dr : mpi.rank_2dc);
		const int size_in_2d = (cq_or_visited ? mpi.size_2dr : mpi.size_2dc);
#if VERTEX_SORTING
		const BitmapType* const visited = (cq_or_visited ? visited_orig_bitmap_ : visited_);
#else
		const BitmapType* const visited = (cq_or_visited ? visited_ : visited_);
#endif
#endif // #ifndef NDEBUG

		// compute number of packets
		for(int i = 0; i < num_src; ++i) {
			int index_start = ((bfs_detail::CompressedStream*)(byte_stream + offset[i]))->packet_index_start;
			packet_count[i] = (count[i] - index_start * sizeof(bfs_detail::PacketIndex) -
					offsetof(bfs_detail::CompressedStream, d)) / sizeof(bfs_detail::PacketIndex);
			packet_offset[i+1] = packet_offset[i] + packet_count[i];
		}

		// make a list of all packets
		LocalVertsIndex num_total_packets = packet_offset[num_src];
		UpdatePacketIndex* index = new UpdatePacketIndex[num_total_packets];
		int64_t num_vertices_received = 0;

#pragma omp parallel if(offset[num_src] > 4096) reduction(+:num_vertices_received)
		{
#pragma omp for
			for(int i = 0; i < num_src; ++i) {
				bfs_detail::CompressedStream* stream = (bfs_detail::CompressedStream*)(byte_stream + offset[i]);
				LocalVertsIndex base = packet_offset[i];
				bfs_detail::PacketIndex* packet_index = &stream->d.index[stream->packet_index_start];
				LocalVertsIndex stream_offset = offset[i] + offsetof(bfs_detail::CompressedStream, d);
#if DEBUG_PRINT
		if(mpi.isMaster()) printf("L:%d:Expand Received Packet (src,num_packet,stream_offset,packet_index_start,stream_size)=(%d,%d,%d,%d,%d)\n", __LINE__,
				i, packet_count[i], stream_offset, stream->packet_index_start, count[i]);
#endif
				for(int64_t k = 0; k < packet_count[i]; ++k) {
					index[base + k].offset = stream_offset;
					index[base + k].length = packet_index[k].length;
					index[base + k].src_num = i;
#ifndef NDEBUG
					index[base + k].num_vertices = packet_index[k].num_int;
#endif
#if DEBUG_PRINT
		if(mpi.isMaster()) printf("L:%d:Packet %"PRId64" (offset,length,num_vertices)=(%d,%d,%d)\n", __LINE__,
				k, stream_offset, packet_index[k].length, packet_index[k].num_int);
#endif
					num_vertices_received += packet_index[k].num_int;
					stream_offset += packet_index[k].length;
				}
				assert((roundup<sizeof(bfs_detail::PacketIndex)>(
						(stream_offset - offset[i] - offsetof(bfs_detail::CompressedStream, d)))
						/ sizeof(bfs_detail::PacketIndex)) == stream->packet_index_start);
			}

			int64_t* decode_buffer = thread_local_buffer_[omp_get_thread_num()]->decode_buffer;

			// update bitmap
#pragma omp for
			for(int64_t i = 0; i < (int64_t)num_total_packets; ++i) {
				int num_vertices = varint_decode_stream_signed(byte_stream + index[i].offset,
						index[i].length, (uint64_t*)decode_buffer);
#ifndef NDEBUG
				assert (num_vertices == index[i].num_vertices);
				int64_t num_local_vertices = get_actual_number_of_local_vertices();
				int64_t dbg_vertices_id = 0;
				for(int64_t r = 0; r < num_vertices; ++r) {
					assert (decode_buffer[r] >= 0);
					assert (decode_buffer[r] < num_local_vertices);
					dbg_vertices_id += decode_buffer[r];
					assert (dbg_vertices_id < num_local_vertices);
				}
#endif
				if(summary) {
					// CQ
					LocalVertsIndex v_swizzled = (LocalVertsIndex(1) << log_local_bitmap_) * NUMBER_PACKING_EDGE_LISTS * index[i].src_num;
					for(int k = 0; k < num_vertices; ++k) {
						v_swizzled += decode_buffer[k];
						LocalVertsIndex word_idx = v_swizzled / NUMBER_PACKING_EDGE_LISTS;
						int bit_idx = v_swizzled % NUMBER_PACKING_EDGE_LISTS;
						BitmapType mask = BitmapType(1) << bit_idx;
#if 0
						const BitmapType fetch_result = target[word_idx];
#pragma omp atomic
						target[word_idx] |= mask;
#else
						const BitmapType fetch_result = __sync_fetch_and_or(&target[word_idx], mask);
#endif
						if(fetch_result == 0) {
							const int64_t bit_offset = word_idx / NUMBER_CQ_SUMMARIZING;
							const int64_t summary_word_idx = bit_offset / (sizeof(summary[0]) * 8);
							const int64_t summary_bit_idx = bit_offset % (sizeof(summary[0]) * 8);
							BitmapType summary_mask = BitmapType(1) << summary_bit_idx;

							if((summary[summary_word_idx] & summary_mask) == 0) {
#if 0
#pragma omp atomic
								summary[summary_word_idx] |= summary_mask;
#else
								__sync_fetch_and_or(&summary[summary_word_idx], summary_mask);
#endif
							}
						}
					}
				}
				else {
					// shared visited
#if SHARED_VISITED_STRIPE
					//const int log_summarizing_verts = LOG_CQ_SUMMARIZING + LOG_PACKING_EDGE_LISTS;
					const LocalVertsIndex src_num_factor = LocalVertsIndex(index[i].src_num)*NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS;
					const int log_size_c = get_msb_index(mpi.size_2dc);
					const LocalVertsIndex mask2 = LocalVertsIndex(NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS) - 1;
					const LocalVertsIndex mask1 = get_number_of_local_vertices() - 1 - mask2;
					LocalVertsIndex v_swizzled = 0;
#else
					LocalVertsIndex v_swizzled = (LocalVertsIndex(1) << log_local_bitmap_) * NUMBER_PACKING_EDGE_LISTS * index[i].src_num;
#endif

					for(int k = 0; k < num_vertices; ++k) {
						v_swizzled += decode_buffer[k];
#if SHARED_VISITED_STRIPE
						LocalVertsIndex sv_idx = src_num_factor | ((v_swizzled & mask1) << log_size_c) | (v_swizzled & mask2);
#else
						LocalVertsIndex sv_idx = v_swizzled;
#endif
						LocalVertsIndex word_idx = sv_idx / NUMBER_PACKING_EDGE_LISTS;
						int bit_idx = sv_idx % NUMBER_PACKING_EDGE_LISTS;
						BitmapType mask = BitmapType(1) << bit_idx;
				//		if((target[word_idx] & mask) == 0) {
#if 0
#pragma omp atomic
							target[word_idx] |= mask;
#else
							__sync_fetch_and_or(&target[word_idx], mask);
#endif
#ifndef NDEBUG
							LocalVertsIndex base = LocalVertsIndex(1) << log_local_bitmap_;
							if(index[i].src_num == rank_in_2d) {
								LocalVertsIndex local_word_idx = word_idx - base * index[i].src_num;
								assert((visited[local_word_idx] & mask) != 0);
							}
							else {
								assert ((word_idx < (base * size_in_2d)) || (word_idx >= (base * (size_in_2d + 1))));
							}
#endif
				//		}
					}
				}
			}
		} // #pragma omp parallel

		delete [] index;
		return num_vertices_received;
	}
#else // #if BFS_EXPAND_COMPRESSION
	// return number of vertices received
	int64_t update_from_stream(uint32_t* recv_data, int* offset, int* count,
			int num_src, BitmapType* target, LocalVertsIndex length, BitmapType* summary, bool cq_or_visited /* for debug */)
	{
#ifndef NDEBUG
		int64_t num_local_vertices = get_actual_number_of_local_vertices();
		const int rank_in_2d = (cq_or_visited ? mpi.rank_2dr : mpi.rank_2dc);
		const int size_in_2d = (cq_or_visited ? mpi.size_2dr : mpi.size_2dc);
#if VERTEX_SORTING
		const BitmapType* const visited = (cq_or_visited ? visited_orig_bitmap_ : visited_);
#else
		const BitmapType* const visited = (cq_or_visited ? visited_ : visited_);
#endif
#endif // #ifndef NDEBUG
		int64_t num_vertices_received = 0;

#pragma omp parallel if(offset[num_src] > 4096) reduction(+:num_vertices_received)
		{
#pragma omp for
			for(int i = 0; i < num_src; ++i) {
				uint32_t* vertices = recv_data + offset[i];
				uint32_t num_vertices = count[i];
#if DEBUG_PRINT
		if(mpi.isMaster()) printf("L:%d:Expand Received Packet (src,num_vertices,vertices_offset)=(%d,%d,%d)\n", __LINE__,
				i, count[i], offset[i]);
#endif


				if(summary) {
					// CQ
					LocalVertsIndex v_swizzled_base = (LocalVertsIndex(1) << log_local_bitmap_) * NUMBER_PACKING_EDGE_LISTS * i;
					for(uint32_t k = 0; k < num_vertices; ++k) {
						assert (vertices[k] < num_local_vertices);
						LocalVertsIndex v_swizzled = v_swizzled_base + vertices[k];
						LocalVertsIndex word_idx = v_swizzled / NUMBER_PACKING_EDGE_LISTS;
						int bit_idx = v_swizzled % NUMBER_PACKING_EDGE_LISTS;
						BitmapType mask = BitmapType(1) << bit_idx;
						const BitmapType old_value = target[word_idx];

						target[word_idx] = old_value | mask; // write

						if(old_value == 0) {
							const int64_t bit_offset = word_idx / NUMBER_CQ_SUMMARIZING;
							const int64_t summary_word_idx = bit_offset / (sizeof(summary[0]) * 8);
							const int64_t summary_bit_idx = bit_offset % (sizeof(summary[0]) * 8);
							BitmapType summary_mask = BitmapType(1) << summary_bit_idx;

							summary[summary_word_idx] |= summary_mask; // write
						}
					}
				}
				else {
					// shared visited
#if SHARED_VISITED_STRIPE
					//const int log_summarizing_verts = LOG_CQ_SUMMARIZING + LOG_PACKING_EDGE_LISTS;
					const LocalVertsIndex src_num_factor = LocalVertsIndex(i)*NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS;
					const int log_size_c = get_msb_index(mpi.size_2dc);
					const LocalVertsIndex mask2 = LocalVertsIndex(NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS) - 1;
					const LocalVertsIndex mask1 = get_number_of_local_vertices() - 1 - mask2;
					LocalVertsIndex v_swizzled_base = 0;
#else
					LocalVertsIndex v_swizzled_base = (LocalVertsIndex(1) << log_local_bitmap_) * NUMBER_PACKING_EDGE_LISTS * i;
#endif

					for(uint32_t k = 0; k < num_vertices; ++k) {
						assert (vertices[k] < num_local_vertices);
						LocalVertsIndex v_swizzled = v_swizzled_base + vertices[k];
#if SHARED_VISITED_STRIPE
						LocalVertsIndex sv_idx = src_num_factor | ((v_swizzled & mask1) << log_size_c) | (v_swizzled & mask2);
#else
						LocalVertsIndex sv_idx = v_swizzled;
#endif
						LocalVertsIndex word_idx = sv_idx / NUMBER_PACKING_EDGE_LISTS;
						int bit_idx = sv_idx % NUMBER_PACKING_EDGE_LISTS;
						BitmapType mask = BitmapType(1) << bit_idx;

						target[word_idx] |= mask; // write

#ifndef NDEBUG
						LocalVertsIndex base = LocalVertsIndex(1) << log_local_bitmap_;
						if(i == rank_in_2d) {
							LocalVertsIndex local_word_idx = word_idx - base * i;
							assert((visited[local_word_idx] & mask) != 0);
						}
						else {
							assert ((word_idx < (base * size_in_2d)) || (word_idx >= (base * (size_in_2d + 1))));
						}
#endif
					}
				}
			}
		} // #pragma omp parallel

		return num_vertices_received;
	}
#endif // #if BFS_EXPAND_COMPRESSION

	void update_shared_visited_from_bitmap(BitmapType* source)
	{
		// shared visited
		BitmapType* shared_visited = this->shared_visited_;
	//	const int64_t bitmap_size = this->get_bitmap_size_v1();
		const int64_t local_bitmap_size_in_lines = get_bitmap_size_visited() >> LOG_CQ_SUMMARIZING;

#pragma omp parallel for
		for(int i = 0; i < local_bitmap_size_in_lines; ++i) {
			for(int k = 0; k < mpi.size_2dc; ++k) {
				// copy cache line
				BitmapType* src = source + NUMBER_CQ_SUMMARIZING * (local_bitmap_size_in_lines * k + i);
				BitmapType* dst = shared_visited + NUMBER_CQ_SUMMARIZING * (mpi.size_2dc * i + k);
				for(int r = 0; r < NUMBER_CQ_SUMMARIZING; ++r) {
					dst[r] = src[r];
				}
			}
		}
	}

	void expand(int64_t global_nq_vertices, int64_t global_visited_vertices, ExpandCommCommand* ex_cq, ExpandCommCommand* ex_vi)
	{
		int64_t global_bitmap_size_in_bytes =
				(int64_t(1) << (graph_.log_global_verts() - LOG_PACKING_EDGE_LISTS)) * sizeof(BitmapType);
		int64_t threshold = int64_t((double)global_bitmap_size_in_bytes *
				(double)EXPAND_COMM_THRESOLD_AVG / (double)EXPAND_COMM_DENOMINATOR);
		int64_t local_data_size;
		EXPAND_PHASE ex_phase = forward_or_backward_ ? EXPAND_FORWARD : EXPAND_BACKWARD;


#if BFS_BACKWARD
		if(forward_or_backward_) {
			const int64_t threashold =
					int64_t((int64_t(1) << graph_.log_actual_global_verts()) *
					((double)BFS_PARAMS::BACKWARD_THREASOLD / (double)BFS_PARAMS::BACKEARD_DENOMINATOR));
			if(global_visited_vertices > threashold) {
				// change direction
				forward_or_backward_ = false;
				ex_phase = EXPAND_SWITCH;
				// fill 1 to summary
				memset(cq_summary_, 0, sizeof(BitmapType) * get_summary_size_v0());
				d_->num_active_vertices_ = int64_t(1) << graph_.log_local_v1();
#if VERVOSE_MODE
			//	if(mpi.isMaster()) fprintf(IMD_OUT, "forward -> backward\n");
#endif
			}
		}
#endif

		fiber_man_.begin_processing();
		if(global_nq_vertices > threshold) {
#if VERVOSE_MODE
			if(mpi.isMaster()) fprintf(IMD_OUT, "Expand using Bitmap.\n");
#endif
			ex_cq->start_comm(0, true, ex_phase, false);
			ex_vi->start_comm(0, true, ex_phase, true);
		}
		else {
			if(ex_phase == EXPAND_SWITCH) {
				expand_create_stream(visited_, ex_cq->local_buffer_, &local_data_size, true);
			} else {
				expand_create_stream(nq_bitmap_, ex_cq->local_buffer_, &local_data_size, true);
			}
			ex_cq->start_comm(local_data_size, false, ex_phase, false);
#if VERTEX_SORTING
			expand_create_stream(nq_sorted_bitmap_, ex_vi->local_buffer_, &local_data_size, false);
#endif
			ex_vi->start_comm(local_data_size, false, ex_phase, true);
		}
#if AVOID_BUSY_WAIT
		bool complete = false;
		const int max_threads = omp_get_max_threads();
		omp_set_num_threads(max_threads - 1);
#pragma omp parallel if(max_threads > 1), num_threads(2)
		{
			// force one thread waiting
			pthread_mutex_lock(&d_->avoid_busy_wait_sync_);
			if(complete == false) {
				fiber_man_.enter_processing();
				complete = true;
			}
			pthread_mutex_unlock(&d_->avoid_busy_wait_sync_);
		} // #pragma omp parallel if(max_threads > 1), num_threads(2)
		omp_set_num_threads(max_threads);
#else // #if AVOID_BUSY_WAIT
		fiber_man_.enter_processing();
#endif // #if AVOID_BUSY_WAIT
	}

	void expand_root(int64_t root_local, ExpandCommCommand* ex_cq, ExpandCommCommand* ex_vi)
	{
		using namespace BFS_PARAMS;
		fiber_man_.begin_processing();
		if(root_local != -1) {
#if VERTEX_SORTING
			int64_t sortd_root_local = graph_.vertex_mapping_[root_local];
#endif

#if BFS_EXPAND_COMPRESSION
			int stream_length = varint_encode(&root_local, 1, ex_cq->local_buffer_->d.stream, bfs_detail::VARINT_EXPAND_CQ);
			int packet_index_start = get_blocks<sizeof(bfs_detail::PacketIndex)>(stream_length);
			ex_cq->local_buffer_->packet_index_start = packet_index_start;
			ex_cq->local_buffer_->d.index[packet_index_start].length = stream_length;
			ex_cq->local_buffer_->d.index[packet_index_start].num_int = 1;
			LocalVertsIndex cq_stream_size = get_compressed_stream_length(1, stream_length);
#else // #if BFS_EXPAND_COMPRESSION
			ex_cq->local_buffer_[0] = root_local;
			LocalVertsIndex cq_stream_size = 1;
#endif // #if BFS_EXPAND_COMPRESSION
			ex_cq->start_comm(cq_stream_size, false, EXPAND_FORWARD, false);
#if DEBUG_PRINT
			if(mpi.isMaster()) printf("L:%d:SPacket (offset,length,num_vertices)=(%d,%d,%d)\n", __LINE__,
					0, ex_cq->local_buffer_->d.index[packet_index_start].length, ex_cq->local_buffer_->d.index[packet_index_start].num_int);
#endif
#if DEBUG_PRINT
			if(mpi.isMaster()) printf("L:%d:Expand Send Packet (num_packet,stream_offset,packet_index_start,stream_size)=(%d,%d,%d,%d)\n", __LINE__,
					1, stream_length, packet_index_start, cq_stream_size);
#endif

#if VERTEX_SORTING
#if BFS_EXPAND_COMPRESSION
			stream_length = varint_encode(&sortd_root_local, 1, ex_vi->local_buffer_->d.stream, bfs_detail::VARINT_EXPAND_SV);
			packet_index_start = get_blocks<sizeof(bfs_detail::PacketIndex)>(stream_length);
			ex_vi->local_buffer_->packet_index_start = packet_index_start;
			ex_vi->local_buffer_->d.index[packet_index_start].length = stream_length;
			ex_vi->local_buffer_->d.index[packet_index_start].num_int = 1;
			LocalVertsIndex vi_stream_size = get_compressed_stream_length(1, stream_length);
#else // #if BFS_EXPAND_COMPRESSION
			ex_vi->local_buffer_[0] = sortd_root_local;
			LocalVertsIndex vi_stream_size = 1;
#endif // #if BFS_EXPAND_COMPRESSION
			ex_vi->start_comm(vi_stream_size, false, EXPAND_FORWARD, true);
#else // #if VERTEX_SORTING
			ex_vi->start_comm(cq_stream_size, false, EXPAND_FORWARD, true);
#endif // #if VERTEX_SORTING
		}
		else {
#if BFS_EXPAND_COMPRESSION
			ex_cq->local_buffer_->packet_index_start = 0;
#if VERTEX_SORTING
			ex_vi->local_buffer_->packet_index_start = 0;
#endif // #if VERTEX_SORTING
			LocalVertsIndex stream_size = get_compressed_stream_length(0, 0);
#else // #if BFS_EXPAND_COMPRESSION
			LocalVertsIndex stream_size = 0;
#endif // #if BFS_EXPAND_COMPRESSION
			ex_cq->start_comm(stream_size, false, EXPAND_FORWARD, false);
			ex_vi->start_comm(stream_size, false, EXPAND_FORWARD, true);
		}
		fiber_man_.enter_processing();
	}

	void fold_send_packet(FoldPacket* packet, int dest_c)
	{
#if VLQ_COMPRESSION
		for(int r = packet->num_edges - 1; r > 0; --r) {
			packet->v0_list[r] -= packet->v0_list[r-1];
			assert(packet->v0_list[r] >= 0);
		}
#if 0
		printf("num_edges = %d\n", packet->num_edges);
#endif
		uint8_t* stream_buffer = thread_local_buffer_[omp_get_thread_num()]->stream_buffer;
		int stream_length = varint_encode(packet->v0_list, packet->num_edges, stream_buffer, bfs_detail::VARINT_FOLD);
		comm_.fold_send(stream_buffer, packet->v1_list, stream_length, packet->num_edges, dest_c);
#else
		comm_.fold_send(packet->edges, packet->num_edges, dest_c);
#endif
	}

#if LOW_LEVEL_FUNCTION
	struct ExtractEdge : public Runnable {
		virtual void run() {
			using namespace BFS_PARAMS;
#if PROFILING_MODE
			ts_commit_.reset();
			profiling::TimeKeeper tk_all;
#endif
			/*
			BitmapType* cq_bitmap = this_->cq_bitmap_;
			BitmapType* cq_summary = this_->cq_summary_;
			BitmapType* shared_visited = this_->shared_visited_;
			BitmapType* bv_bitmap = this_->graph_.bv_bitmap_;
			const int64_t* row_starts = this_->graph_.row_starts_;
			const IndexArray& index_array = this_->graph_.index_array_;
			const int log_local_verts = this_->graph_.log_local_verts();
			const int64_t local_verts_mask = this_->get_number_of_local_vertices() - 1;
			int64_t i_start = i_start_, i_end = i_end_;
			FoldPacket* packet_array =
					this_->thread_local_buffer_[omp_get_thread_num()]->fold_packet;
			int64_t num_edge_relax = 0;
			*/
			ThreadLocalBuffer* tlb = this_->thread_local_buffer_[omp_get_thread_num()];
		//	const int log_local_verts = this_->graph_.log_local_verts();
			BitmapType* const cq_summary = this_->cq_summary_;

			// TODO: optimize
			params.cq_bitmap = this_->cq_bitmap_;
			params.cq_summary = cq_summary;
			params.visited_bitmap = this_->shared_visited_;
			params.bv_bitmap = this_->graph_.bv_bitmap_;
			params.bv_offsets = this_->graph_.bv_offsets_;
			params.bv_row_starts = this_->graph_.bv_row_starts_;
#if EDGES_IN_RAIL
			params.idx_array = this_->graph_.index_array_.get_ptr();
#else
			params.idx_high = this_->graph_.index_array_.get_ptr_high();
			params.idx_low = this_->graph_.index_array_.get_ptr_low();
#endif
#if BIT_SCAN_TABLE
			params.bit_scan_table = this_->bit_scan_table_;
#endif
			params.v0_high_c = (int64_t(mpi.rank_2dc) << this_->graph_.log_local_v0());
			params.local_verts_mask = this_->get_number_of_local_vertices() - 1;
			params.log_local_verts = this_->graph_.log_local_verts();
		//	params.log_r = get_msb_index(mpi.size_2dr);
		//	params.swizzed_r = (int64_t(mpi.rank_2dr) << log_local_verts);
#if VERVOSE_MODE
			params.num_edge_relax = 0;
#endif
		//	params.num_active_vertices = 0;
			params.submit_packetf = submit_packet_;
			params.context = this;
			params.row_array = tlb->row_buffer;
#if PROFILING_MODE
			params.time_vertex_scan = 0;
#endif

			packet_array_ = tlb->fold_packet;

			int64_t i_start = i_start_, i_end = i_end_;
			for(int64_t i = i_start; i < i_end; ++i) {
				BitmapType summary_i = cq_summary[i];
				if(summary_i == 0) continue;
				params.cur_summary = summary_i;
				params.summary_idx = i;

				bfs_forward_extract_edge(packet_array_, &params);

				// write zero after read
				cq_summary[i] = 0;
			} // for(int64_t i = i_start; i < i_end; ++i) {
			//
#if PROFILING_MODE
			profiling::TimeSpan ts_all;
			ts_all += tk_all;
			ts_all -= ts_commit_;
			ts_all -= params.time_vertex_scan;
			this_->extract_edge_time_ += ts_all;
			this_->commit_time_ += ts_commit_;
			this_->vertex_scan_time_ += params.time_vertex_scan;
#endif
			volatile int* jobs_ptr = &this_->d_->num_remaining_extract_jobs_;
			if(__sync_fetch_and_add(jobs_ptr, -1) == 1) {
				this_->fiber_man_.submit_array(this_->sched_.fold_end_job, mpi.size_2dc, 0);
			}
#if VERVOSE_MODE
			__sync_fetch_and_add(&this_->d_->num_edge_relax_, params.num_edge_relax);
#endif
		}
		static void submit_packet_(void* this__, int dest_c) {
			static_cast<ExtractEdge*>(this__)->internal_submit_packet(dest_c);
		}
		void internal_submit_packet(int dest_c) {
#if PROFILING_MODE
			profiling::TimeKeeper tk_commit;
#endif
			this_->fold_send_packet(&packet_array_[dest_c], dest_c);
#if PROFILING_MODE
			ts_commit_ += tk_commit;
#endif
		}
		FoldPacket* packet_array_;
		BfsExtractContext params;
		ThisType* this_;
		int64_t i_start_, i_end_;
#if PROFILING_MODE
		profiling::TimeSpan ts_commit_;
#endif
	};

#else // #if LOW_LEVEL_FUNCTION

	struct ExtractEdge : public Runnable {
		virtual void run() {
			using namespace BFS_PARAMS;
#if PROFILING_MODE
			profiling::TimeKeeper tk_all;
			profiling::TimeKeeper tk_commit;
			profiling::TimeSpan ts_commit;
#endif
			BitmapType* cq_bitmap = this_->cq_bitmap_;
			BitmapType* cq_summary = this_->cq_summary_;
			BitmapType* shared_visited = this_->shared_visited_;
#if GRAPH_BITVECTOR
			BitmapType* bv_bitmap = this_->graph_.bv_bitmap_;
#endif
			const int64_t* row_starts = this_->graph_.row_starts_;
			const IndexArray& index_array = this_->graph_.index_array_;
			const int log_local_verts = this_->graph_.log_local_verts();
			const int64_t local_verts_mask = this_->get_number_of_local_vertices() - 1;
#if VERTEX_SORTING
			const int64_t threshold = this_->get_number_of_local_vertices() / BFS_PARAMS::DENOM_SHARED_VISITED_PART;
#endif // #if VERTEX_SORTING
			int64_t i_start = i_start_, i_end = i_end_;
			FoldPacket* packet_array =
					this_->thread_local_buffer_[omp_get_thread_num()]->fold_packet;

#if SHARED_VISITED_STRIPE
			const int log_size_c = get_msb_index(mpi.size_2dc);
			const int64_t mask2 = LocalVertsIndex(NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS) - 1;
			const int64_t mask1 = this_->get_number_of_local_vertices() - 1 - mask2;
#endif
#if VERVOSE_MODE
			int64_t num_edge_relax = 0;
#endif

			for(int64_t i = i_start; i < i_end; ++i) {
				BitmapType summary_i = cq_summary[i];
				if(summary_i == 0) continue;
				for(int ii = 0; ii < (int)sizeof(cq_summary[0])*8; ++ii) {
					if(summary_i & (BitmapType(1) << ii)) {
						int64_t cq_base_offset = ((int64_t)i*sizeof(cq_summary[0])*8 + ii)*NUMBER_CQ_SUMMARIZING;
						for(int k = 0; k < NUMBER_CQ_SUMMARIZING; ++k) {
							int64_t e0 = cq_base_offset + k;
#if GRAPH_BITVECTOR
							BitmapType bitmap_k = cq_bitmap[e0] & bv_bitmap[e0];
#else
							BitmapType bitmap_k = cq_bitmap[e0];
#endif
							if(bitmap_k == 0) {
#if GRAPH_BITVECTOR
								// write zero after read
								cq_bitmap[e0] = 0;
#endif
								continue;
							}
							// TODO: this_->graph_.log_local_v0()
							int64_t v0_high = (e0 << LOG_PACKING_EDGE_LISTS) |
									(int64_t(mpi.rank_2dc) << this_->graph_.log_local_v0());
#if BFS_FORWARD_PREFETCH // for FUJITSU
							int64_t tmp_index[128];
							int64_t tmp_word_idx[128];
							BitmapType tmp_mask[128];
							for(int64_t ri = row_starts[e0]; ri < row_starts[e0+1]; ri += 128) {
								int64_t ri_end = std::min(ri + 128, row_starts[e0+1]);
								int num_tmp_edges = 0;
								for(int64_t rii = ri; rii < ri_end; ++rii) {
									int8_t row_lowbits = (int8_t)(index_array.low_bits(rii) % NUMBER_PACKING_EDGE_LISTS);
									if ((bitmap_k & (int64_t(1) << row_lowbits)) != 0){
										int64_t index = index_array(rii);
										int64_t c1 = index >> LOG_PACKING_EDGE_LISTS;
#if VERTEX_SORTING
										int64_t v1_local = c1 & local_verts_mask;
#if SHARED_VISITED_OPT
										if(v1_local < threshold) {
											tmp_index[num_tmp_edges] = index;
											tmp_word_idx[num_tmp_edges] = c1 / NUMBER_PACKING_EDGE_LISTS;
											int bit_idx = c1 % NUMBER_PACKING_EDGE_LISTS;
											tmp_mask[num_tmp_edges] = BitmapType(1) << bit_idx;
											++num_tmp_edges;
										}
										else
#endif
										{
											int64_t dest_c = c1 >> log_local_verts;
											int64_t v0_swizzled = v0_high | row_lowbits;

											FoldPacket* packet = &packet_array[dest_c];
											packet->v0_list[packet->num_edges] = v0_swizzled;
											packet->v1_list[packet->num_edges] = v1_local;
											if(++packet->num_edges == PACKET_LENGTH) {
#if PROFILING_MODE
												tk_commit.getSpanAndReset();
#endif
												this_->fold_send_packet(packet, dest_c);
#if PROFILING_MODE
												ts_commit += tk_commit;
#endif
												packet->num_edges = 0;
											}
										}
#else // #if VERTEX_SORTING
										tmp_index[num_tmp_edges] = index;
										tmp_word_idx[num_tmp_edges] = c1 / NUMBER_PACKING_EDGE_LISTS;
										int bit_idx = c1 % NUMBER_PACKING_EDGE_LISTS;
										tmp_mask[num_tmp_edges] = BitmapType(1) << bit_idx;
										++num_tmp_edges;
#endif // #if VERTEX_SORTING
									}
								}
#ifdef __FUJITSU
#pragma loop prefetch
#endif
								for(int e = 0; e < num_tmp_edges; ++e) {
									int64_t word_idx = tmp_word_idx[e];
									BitmapType mask = tmp_mask[e];
									if((shared_visited[word_idx] & mask) ||
									 (__sync_fetch_and_or(&shared_visited[word_idx], mask) & mask))
									{
										continue;
									}

									int8_t row_lowbits = tmp_index[e] % NUMBER_PACKING_EDGE_LISTS;
									int64_t c1 = tmp_index[e] >> LOG_PACKING_EDGE_LISTS;
									int64_t v1_local = c1 & local_verts_mask;
									int64_t dest_c = c1 >> log_local_verts;
									int64_t v0_swizzled = v0_high | row_lowbits;

									FoldPacket* packet = &packet_array[dest_c];
									packet->v0_list[packet->num_edges] = v0_swizzled;
									packet->v1_list[packet->num_edges] = v1_local;
									if(++packet->num_edges == PACKET_LENGTH) {
#if PROFILING_MODE
										tk_commit.getSpanAndReset();
#endif
										this_->fold_send_packet(packet, dest_c);
#if PROFILING_MODE
										ts_commit += tk_commit;
#endif
										packet->num_edges = 0;
									}
								}
#if VERVOSE_MODE
								num_edge_relax += num_tmp_edges;
#endif
							}
#else
							for(int64_t ri = row_starts[e0]; ri < row_starts[e0+1]; ++ri) {
								int8_t row_lowbits = (int8_t)(index_array.low_bits(ri) % NUMBER_PACKING_EDGE_LISTS);
								if ((bitmap_k & (int64_t(1) << row_lowbits)) != 0){
									int64_t c1 = index_array(ri) >> LOG_PACKING_EDGE_LISTS;
									int64_t v1_local = c1 & local_verts_mask;
									int64_t dest_c = c1 >> log_local_verts;
#if SHARED_VISITED_OPT
									if(v1_local < threshold) {
										// --- new algorithm begin ---
#if SHARED_VISITED_STRIPE
										int64_t sv_idx = (dest_c*NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS) |
												((c1 & mask1) << log_size_c) | (c1 & mask2);
#else
										int64_t sv_idx = c1;
#endif
										int64_t word_idx = sv_idx / NUMBER_PACKING_EDGE_LISTS;
										int bit_idx = sv_idx % NUMBER_PACKING_EDGE_LISTS;
										BitmapType mask = BitmapType(1) << bit_idx;

#if 0
										if(shared_visited[word_idx] & mask) {
#pragma omp atomic
											shared_visited[word_idx] |= mask;
											continue;
										}
#else
										if((shared_visited[word_idx] & mask) ||
										 (__sync_fetch_and_or(&shared_visited[word_idx], mask) & mask))
										{
											continue;
										}
#endif
										// --- new algorithm end ---
									}
#endif
									// TODO:
									int64_t v0_swizzled = v0_high | row_lowbits;

									FoldPacket* packet = &packet_array[dest_c];
									packet->v0_list[packet->num_edges] = v0_swizzled;
									packet->v1_list[packet->num_edges] = v1_local;
									if(++packet->num_edges == PACKET_LENGTH) {
#if PROFILING_MODE
										tk_commit.getSpanAndReset();
#endif
										this_->fold_send_packet(packet, dest_c);
#if PROFILING_MODE
										ts_commit += tk_commit;
#endif
										packet->num_edges = 0;
									}
								}
							}
#endif
							// write zero after read
							cq_bitmap[e0] = 0;
						}
					}
				}
				// write zero after read
				cq_summary[i] = 0;
			}
			//
#if PROFILING_MODE
			profiling::TimeSpan ts_all;
			ts_all += tk_all;
			ts_all -= ts_commit;
			this_->extract_edge_time_ += ts_all;
			this_->commit_time_ += ts_commit;
#endif
			volatile int* jobs_ptr = &this_->d_->num_remaining_extract_jobs_;
			if(__sync_fetch_and_add(jobs_ptr, -1) == 1) {
				this_->fiber_man_.submit_array(this_->sched_.fold_end_job, mpi.size_2dc, 0);
			}
#if VERVOSE_MODE
			__sync_fetch_and_add(&this_->d_->num_edge_relax_, num_edge_relax);
#endif
		}
		ThisType* this_;
		int64_t i_start_, i_end_;
	};

#endif // #if LOW_LEVEL_FUNCTION

#if LOW_LEVEL_FUNCTION
	struct BackwardExtractEdge : public Runnable {
		virtual void run() {
			using namespace BFS_PARAMS;
#if PROFILING_MODE
			ts_commit_.reset();
			profiling::TimeKeeper tk_all;
#endif
			/*
			// convert name so that it properly expresses containing data
			BitmapType* const visited_bitmap = this_->cq_bitmap_;
			BitmapType* const visited_summary = this_->cq_summary_;
			BitmapType* const cq_bitmap = this_->shared_visited_;
			BitmapType* const bv_bitmap = this_->graph_.bv_bitmap_;
			int64_t* const bv_offsets = this_->graph_.bv_offsets_;
			int64_t* const bv_row_starts = this_->graph_.bv_row_starts_;
			const IndexArray& index_array = this_->graph_.index_array_;
			const int log_local_verts = this_->graph_.log_local_verts();
			const int64_t local_verts_mask = this_->get_number_of_local_vertices() - 1;
			const int log_r = get_msb_index(mpi.size_2dr);
			const int64_t swizzed_r = (int64_t(mpi.rank_2dr) << log_local_verts);
		//	const int64_t threshold = this_->get_number_of_local_vertices() / BFS_PARAMS::DENOM_SHARED_VISITED_PART;
			int64_t i_start = i_start_, i_end = i_end_;
			*/
			ThreadLocalBuffer* tlb = this_->thread_local_buffer_[omp_get_thread_num()];
			const int log_local_verts = this_->graph_.log_local_verts();
			BitmapType* const visited_summary = this_->cq_summary_;

			// TODO: optimize
			params.visited_bitmap = this_->cq_bitmap_;
			params.visited_summary = visited_summary;
			params.cq_bitmap = this_->shared_visited_;
			params.bv_bitmap = this_->graph_.bv_bitmap_;
			params.bv_offsets = this_->graph_.bv_offsets_;
			params.bv_row_starts = this_->graph_.bv_row_starts_;
#if EDGES_IN_RAIL
			params.idx_array = this_->graph_.index_array_.get_ptr();
#else
			params.idx_high = this_->graph_.index_array_.get_ptr_high();
			params.idx_low = this_->graph_.index_array_.get_ptr_low();
#endif
#if BIT_SCAN_TABLE
			params.bit_scan_table = this_->bit_scan_table_;
#endif
		//	params.v0_high_c = (int64_t(mpi.rank_2dc) << this_->graph_.log_local_v0());
			params.local_verts_mask = this_->get_number_of_local_vertices() - 1;
		//	params.log_local_verts = this_->graph_.log_local_verts();
			params.log_r = get_msb_index(mpi.size_2dr);
			params.swizzed_r = (int64_t(mpi.rank_2dr) << log_local_verts);
#if VERVOSE_MODE
			params.num_edge_relax = 0;
#endif
			params.num_active_vertices = 0;
			params.submit_packetb = submit_packet_;
			params.context = this;
			params.row_array = tlb->row_buffer;
#if PROFILING_MODE
			params.time_vertex_scan = 0;
			params.time_core_proc = 0;
#endif

			int64_t i_start = i_start_, i_end = i_end_;
			for(int64_t i = i_start; i < i_end; ++i) {
				BitmapType summary_i = visited_summary[i];
				if(summary_i == BitmapType(-1)) continue;
				const int dest_r = (i*64*NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS) >> log_local_verts;
				assert (dest_r < mpi.size_2dr);
				current_dest_r_ = dest_r;
				current_packet_ = &tlb->fold_packet[dest_r];
				params.cur_summary = summary_i;
				params.summary_idx = i;

				bfs_backward_extract_edge(current_packet_, &params);

			} // for(int64_t i = i_start; i < i_end; ++i) {
			//
#if PROFILING_MODE
			profiling::TimeSpan ts_all;
			ts_all += tk_all;
			ts_all -= ts_commit_;
			ts_all -= params.time_vertex_scan;
			this_->extract_edge_time_ += ts_all;
			this_->commit_time_ += ts_commit_;
			this_->vertex_scan_time_ += params.time_vertex_scan;
#if DETAILED_PROF_MODE
			this_->core_proc_time_ += params.time_core_proc;
#endif
#endif
			volatile int* jobs_ptr = &this_->d_->num_remaining_extract_jobs_;
			if(__sync_fetch_and_add(jobs_ptr, -1) == 1) {
				this_->fiber_man_.submit_array(this_->sched_.fold_end_job, mpi.size_2dr, 0);
			}
			__sync_fetch_and_add(&this_->d_->num_active_vertices_, params.num_active_vertices);
#if VERVOSE_MODE
			__sync_fetch_and_add(&this_->d_->num_edge_relax_, params.num_edge_relax);
#endif
		}
		static void submit_packet_(void* this__) {
			static_cast<BackwardExtractEdge*>(this__)->internal_submit_packet();
		}
		void internal_submit_packet() {
#if PROFILING_MODE
			profiling::TimeKeeper tk_commit;
#endif
			this_->fold_send_packet(current_packet_, current_dest_r_);
#if PROFILING_MODE
			ts_commit_ += tk_commit;
#endif
		}
		int current_dest_r_;
		FoldPacket* current_packet_;
		BfsExtractContext params;
		ThisType* this_;
		int64_t i_start_, i_end_;
#if PROFILING_MODE
		profiling::TimeSpan ts_commit_;
#endif
	};

#else // #if LOW_LEVEL_FUNCTION

	struct BackwardExtractEdge : public Runnable {
		virtual void run() {
			using namespace BFS_PARAMS;
#if PROFILING_MODE
			profiling::TimeKeeper tk_all;
			profiling::TimeKeeper tk_commit;
			profiling::TimeSpan ts_commit;
#endif
			// convert name so that it properly expresses containing data
			BitmapType* const visited_bitmap = this_->cq_bitmap_;
			BitmapType* const visited_summary = this_->cq_summary_;
			BitmapType* const cq_bitmap = this_->shared_visited_;
#if GRAPH_BITVECTOR
			BitmapType* const bv_bitmap = this_->graph_.bv_bitmap_;
#if GRAPH_BITVECTOR_OFFSET
			int64_t* const bv_offsets = this_->graph_.bv_offsets_;
			int64_t* const bv_row_starts = this_->graph_.bv_row_starts_;
#endif
#endif
			const int64_t* row_starts = this_->graph_.row_starts_;
			const IndexArray& index_array = this_->graph_.index_array_;
			const int log_local_verts = this_->graph_.log_local_verts();
			const int64_t local_verts_mask = this_->get_number_of_local_vertices() - 1;
			const int log_r = get_msb_index(mpi.size_2dr);
			const int64_t swizzed_r = (int64_t(mpi.rank_2dr) << log_local_verts);
		//	const int64_t threshold = this_->get_number_of_local_vertices() / BFS_PARAMS::DENOM_SHARED_VISITED_PART;
			int64_t i_start = i_start_, i_end = i_end_;
			FoldPacket* packet_array =
					this_->thread_local_buffer_[omp_get_thread_num()]->fold_packet;
			int64_t num_active_vertices = 0;
#if VERVOSE_MODE
			int64_t num_edge_relax = 0;
#endif

			for(int64_t i = i_start; i < i_end; ++i) {
				BitmapType summary_i = visited_summary[i];
				if(summary_i == BitmapType(-1)) continue;
				const int dest_r = (i*sizeof(visited_summary[0])*8*NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS) >> log_local_verts;
				assert (dest_r < mpi.size_2dr);
				FoldPacket* const packet = &packet_array[dest_r];
				int pk_num_edges = packet->num_edges;

				for(int ii = 0; ii < (int)sizeof(visited_summary[0])*8; ++ii) {
					if((summary_i & (BitmapType(1) << ii)) == 0) {
						int unvisited_count = 0;
						int64_t cq_base_offset = ((int64_t)i*sizeof(visited_summary[0])*8 + ii)*NUMBER_CQ_SUMMARIZING;
						for(int k = 0; k < NUMBER_CQ_SUMMARIZING; ++k) {
							int64_t e0 = cq_base_offset + k;
#if GRAPH_BITVECTOR
							// Ignore vertices whose edges we do not have.
							BitmapType bitmap_k = visited_bitmap[e0] | ~(bv_bitmap[e0]);
#else // #if GRAPH_BITVECTOR
							BitmapType bitmap_k = visited_bitmap[e0];
#endif // #if GRAPH_BITVECTOR
							if(bitmap_k == BitmapType(-1)) continue;

							++unvisited_count;
							const int64_t v1_high = (e0 * NUMBER_PACKING_EDGE_LISTS) & local_verts_mask;
#if BFS_BACKWARD_PREFETCH // prefetch
							const int64_t start = row_starts[e0];
							const int64_t end = row_starts[e0+1];
							for(int64_t ri = start - BACKWARD_PREFETCH_LENGTH; ri < end; ++ri) {
								if(ri >= start) {
									// processing
									const int8_t row_lowbits = (int8_t)(index_array.low_bits(ri) % NUMBER_PACKING_EDGE_LISTS);
									if ((bitmap_k & (BitmapType(1) << row_lowbits)) == 0){
										// c0 : local + C
										const int64_t c0 = index_array(ri) >> LOG_PACKING_EDGE_LISTS;
#if SHARED_VISITED_STRIPE
										const int64_t sv_idx = (dest_r*NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS) |
//											((c0 & mask1) << log_size_c) | (c0 & mask2); // ????
#else // #if SHARED_VISITED_STRIPE
										const int64_t sv_idx = c0;
#endif // #if SHARED_VISITED_STRIPE
										const int64_t word_idx = sv_idx / NUMBER_PACKING_EDGE_LISTS;
										const int bit_idx = sv_idx % NUMBER_PACKING_EDGE_LISTS;
										if((cq_bitmap[word_idx] & (BitmapType(1) << bit_idx)) == 0) continue;
										// equivalent to : (c0 & ~local_verts_mask) * mpi.size_2dr
										const int64_t v0_c = (c0 & ~local_verts_mask) << log_r;
										const int64_t v0_swizzled = v0_c | swizzed_r | (c0 & local_verts_mask);
										// (e0 * NUMBER_PACKING_EDGE_LISTS) | row_lowbits : local + R
										const int64_t v1_local = (v1_high | row_lowbits);

										packet->v0_list[pk_num_edges] = v0_swizzled; // sorted
										packet->v1_list[pk_num_edges] = v1_local; // original
										if(++pk_num_edges == PACKET_LENGTH) {
											packet->num_edges = PACKET_LENGTH;
#if PROFILING_MODE
											tk_commit.getSpanAndReset();
#endif
											this_->fold_send_packet(packet, dest_r);
#if PROFILING_MODE
											ts_commit += tk_commit;
#endif
											pk_num_edges = 0;
										}

										bitmap_k |= (BitmapType(1) << row_lowbits);
									} // if ((bitmap_k & (int64_t(1) << row_lowbits)) != 0){
								}
								if(ri + BACKWARD_PREFETCH_LENGTH < end) {
									// prefetch
									const int8_t row_lowbits = (int8_t)(index_array.low_bits(ri + BACKWARD_PREFETCH_LENGTH) % NUMBER_PACKING_EDGE_LISTS);
									if ((bitmap_k & (BitmapType(1) << row_lowbits)) == 0){
										// c0 : local + C
										const int64_t c0 = index_array(ri + BACKWARD_PREFETCH_LENGTH) >> LOG_PACKING_EDGE_LISTS;
#if SHARED_VISITED_STRIPE
										const int64_t sv_idx = (dest_r*NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS) |
//											((c0 & mask1) << log_size_c) | (c0 & mask2); // ????
#else // #if SHARED_VISITED_STRIPE
										const int64_t sv_idx = c0;
#endif // #if SHARED_VISITED_STRIPE
										const int64_t word_idx = sv_idx / NUMBER_PACKING_EDGE_LISTS;

										__builtin_prefetch(&cq_bitmap[word_idx], 0);
									}
								}
							} // for(int64_t ri = row_starts[e0]; ri < row_starts[e0+1]; ++ri) {
#else // #if BFS_BACKWARD_PREFETCH
							// TODO: optimize
#if GRAPH_BITVECTOR_OFFSET
							const BitmapType bv_bits = bv_bitmap[e0];
							const int64_t bv_offset_base = bv_offsets[e0];
							for(int row_lowbits = 0; row_lowbits < NUMBER_PACKING_EDGE_LISTS; ++row_lowbits) {
								const BitmapType i3_bit = (BitmapType(1) << row_lowbits);
								if((bitmap_k & i3_bit) == 0) {
									const BitmapType bv = (bv_bits & (i3_bit - 1));
									const int64_t bv_offset = bv_offset_base + __builtin_popcountl(bv);
									for(int64_t ri = bv_row_starts[bv_offset]; ri < bv_row_starts[bv_offset + 1]; ++ri) {
#else // #if GRAPH_BITVECTOR_OFFSET
							for(int64_t ri = row_starts[e0]; ri < row_starts[e0+1]; ++ri) {
								const int8_t row_lowbits = (int8_t)(index_array.low_bits(ri) % NUMBER_PACKING_EDGE_LISTS);
								if ((bitmap_k & (BitmapType(1) << row_lowbits)) == 0){
#endif // #if GRAPH_BITVECTOR_OFFSET
#if VERVOSE_MODE
									++num_edge_relax;
#endif
									// c0 : local + C
									const int64_t c0 = index_array(ri) >> LOG_PACKING_EDGE_LISTS;
#if SHARED_VISITED_STRIPE
									const int64_t sv_idx = (dest_r*NUMBER_CQ_SUMMARIZING*NUMBER_PACKING_EDGE_LISTS) |
//											((c0 & mask1) << log_size_c) | (c0 & mask2); // ????
#else // #if SHARED_VISITED_STRIPE
									const int64_t sv_idx = c0;
#endif // #if SHARED_VISITED_STRIPE
									const int64_t word_idx = sv_idx / NUMBER_PACKING_EDGE_LISTS;
									const int bit_idx = sv_idx % NUMBER_PACKING_EDGE_LISTS;
									if((cq_bitmap[word_idx] & (BitmapType(1) << bit_idx)) == 0) continue;

									// equivalent to : (c0 & ~local_verts_mask) * mpi.size_2dr
									const int64_t v0_c = (c0 & ~local_verts_mask) << log_r;
									const int64_t v0_swizzled = v0_c | swizzed_r | (c0 & local_verts_mask);
									// (e0 * NUMBER_PACKING_EDGE_LISTS) | row_lowbits : local + R
									const int64_t v1_local = (v1_high | row_lowbits);

									packet->v0_list[pk_num_edges] = v0_swizzled; // sorted
									packet->v1_list[pk_num_edges] = v1_local; // original
									if(++pk_num_edges == PACKET_LENGTH) {
										packet->num_edges = PACKET_LENGTH;
#if PROFILING_MODE
										tk_commit.getSpanAndReset();
#endif
										this_->fold_send_packet(packet, dest_r);
#if PROFILING_MODE
										ts_commit += tk_commit;
#endif
										pk_num_edges = 0;
									}

#if GRAPH_BITVECTOR_OFFSET
									break;
									}
								}
							}
#else // #if GRAPH_BITVECTOR_OFFSET
									bitmap_k |= (BitmapType(1) << row_lowbits);
								} // if ((bitmap_k & (int64_t(1) << row_lowbits)) != 0){
							} // for(int64_t ri = row_starts[e0]; ri < row_starts[e0+1]; ++ri) {
#endif // #if GRAPH_BITVECTOR_OFFSET

							// We do not need to update visited_bitmap because it is updated in expand phase.
							// visited_bitmap[e0] |= bitmap_k;
							num_active_vertices += __builtin_popcountl(~bitmap_k);
#endif // #if BFS_BACKWARD_PREFETCH
						} // for(int k = 0; k < NUMBER_CQ_SUMMARIZING; ++k) {
						if(unvisited_count == 0) {
							// update summary
							visited_summary[i] |= (BitmapType(1) << ii);
						} // if(unvisited_count == 0) {

					} // if(summary_i & (BitmapType(1) << ii)) {
				} // for(int ii = 0; ii < (int)sizeof(visited_summary[0])*8; ++ii) {

				packet->num_edges = pk_num_edges;
			} // for(int64_t i = i_start; i < i_end; ++i) {
			//
#if PROFILING_MODE
			profiling::TimeSpan ts_all;
			ts_all += tk_all;
			ts_all -= ts_commit;
			this_->extract_edge_time_ += ts_all;
			this_->commit_time_ += ts_commit;
#endif
			volatile int* jobs_ptr = &this_->d_->num_remaining_extract_jobs_;
			if(__sync_fetch_and_add(jobs_ptr, -1) == 1) {
				this_->fiber_man_.submit_array(this_->sched_.fold_end_job, mpi.size_2dr, 0);
			}
			__sync_fetch_and_add(&this_->d_->num_active_vertices_, num_active_vertices);
#if VERVOSE_MODE
			__sync_fetch_and_add(&this_->d_->num_edge_relax_, num_edge_relax);
#endif
		}
		ThisType* this_;
		int64_t i_start_, i_end_;
	};

#endif // #if LOW_LEVEL_FUNCTION

	struct ReceiverProcessing : public Runnable {
		ReceiverProcessing(ThisType* this__, bfs_detail::FoldCommBuffer* data__)
			: this_(this__), data_(data__) 	{ }
		virtual void run() {
#if PROFILING_MODE
			profiling::TimeKeeper tk_all;
			profiling::TimeKeeper tk_commit;
#endif
			using namespace BFS_PARAMS;
#if VLQ_COMPRESSION
			int64_t* decode_buffer = tlb->decode_buffer;
			const uint8_t* v0_stream = data_->v0_stream->d.stream;
			const bfs_detail::PacketIndex* packet_index =
					&data_->v0_stream->d.index[data_->v0_stream->packet_index_start];
#endif
			BitmapType* restrict const visited = this_->visited_;
			BitmapType* restrict const nq_bitmap = this_->nq_bitmap_;
#if VERTEX_SORTING
			BitmapType* restrict const nq_sorted_bitmap = this_->nq_sorted_bitmap_;
#endif
			int64_t* restrict const pred = this_->pred_;
			const int cur_level = this_->current_level_;
#if VLQ_COMPRESSION
			int v0_offset = 0, v1_offset = 0;
#endif
			int64_t num_nq_vertices = 0;

			const int log_local_verts = this_->graph_.log_local_verts();
			const int log_size = get_msb_index(mpi.size_2d);
			const int64_t local_verts_mask = this_->get_number_of_local_vertices() - 1;
#if LOW_LEVEL_FUNCTION
			num_nq_vertices = bfs_receiver_processing(
					data_->edges,
					data_->num_edges,
					visited,
					nq_bitmap,
					pred,
					cur_level,
					log_local_verts,
					log_size,
					local_verts_mask);
#else // #if LOW_LEVEL_FUNCTION
#define UNSWIZZLE_VERTEX(c) (((c) >> log_local_verts) | (((c) & local_verts_mask) << log_size))
#if VLQ_COMPRESSION
			for(int i = 0; i < data_->num_packets; ++i) {
				int stream_length = packet_index[i].length;
				int num_edges = packet_index[i].num_int;
#ifndef NDEBUG
				int decoded_elements =
#endif
				varint_decode_stream_signed(&v0_stream[v0_offset],
						stream_length, (uint64_t*)decode_buffer);
#if 0
				if(decoded_elements != num_edges) {
					printf("Error: Decode: decoded_elements = %d, num_edges = %d\n", decoded_elements, num_edges);
				}
#endif
				assert (decoded_elements == num_edges);
				const uint32_t* const v1_list = &data_->v1_list[v1_offset];
				int64_t v0_swizzled = 0;

				for(int c = 0; c < num_edges; ++c) {
					v0_swizzled += decode_buffer[c];
#else
			const int64_t* const v0_list = data_->v0_list;
			const uint32_t* const v1_list = data_->v1_list;
			for(int c = 0; c < data_->num_edges; ++c) {
					int64_t v0_swizzled = v0_list[c];
#endif
					const LocalVertsIndex v1_local = v1_list[c];
					const LocalVertsIndex word_idx = v1_local / NUMBER_PACKING_EDGE_LISTS;
					const int bit_idx = v1_local % NUMBER_PACKING_EDGE_LISTS;
					const BitmapType mask = BitmapType(1) << bit_idx;

					if((visited[word_idx] & mask) == 0) { // if this vertex has not visited
						if((__sync_fetch_and_or(&visited[word_idx], mask) & mask) == 0) {
							const int64_t v0 = UNSWIZZLE_VERTEX(v0_swizzled);
		//					const int64_t pred_v = (v0 & int64_t(0xFFFFFFFFFFFF)) | ((int64_t)cur_level << 48);
							const int64_t pred_v = v0 | ((int64_t)cur_level << 48);
#if VERTEX_SORTING
#if 0
#pragma omp atomic
							nq_sorted_bitmap[word_idx] |= mask;
#else
							__sync_fetch_and_or(&nq_sorted_bitmap[word_idx], mask);
#endif
							const LocalVertsIndex orig_v1_local = this_->graph_.invert_vertex_mapping_[v1_local];
							const LocalVertsIndex orig_word_idx = orig_v1_local / NUMBER_PACKING_EDGE_LISTS;
							const int orig_bit_idx = orig_v1_local % NUMBER_PACKING_EDGE_LISTS;
							const BitmapType orig_mask = BitmapType(1) << orig_bit_idx;
							assert (pred[orig_v1_local] == -1);
#if 0
#pragma omp atomic
							nq_bitmap[orig_word_idx] |= orig_mask;
#else
							__sync_fetch_and_or(&nq_bitmap[orig_word_idx], orig_mask);
#endif
							pred[orig_v1_local] = pred_v;
#else
							assert (pred[v1_local] == -1);
#if 0
#pragma omp atomic
							nq_bitmap[word_idx] |= mask;
#else
							__sync_fetch_and_or(&nq_bitmap[word_idx], mask);
#endif
							pred[v1_local] = pred_v;
#endif
							++num_nq_vertices;
						}
					}
#if VLQ_COMPRESSION
				}
				v0_offset += stream_length;
				v1_offset += num_edges;
#endif
			}
#undef UNSWIZZLE_VERTEX
#endif // #if LOW_LEVEL_FUNCTION
			this_->comm_.relase_fold_buffer(data_);
			__sync_fetch_and_add(&this_->d_->num_vertices_in_nq_, num_nq_vertices);
#if PROFILING_MODE
			this_->recv_proc_time_ += tk_all;
#endif
#if 0
			this_->recv_task_.push(this);
#else
			delete this;
#endif
		}
		ThisType* const this_;
		bfs_detail::FoldCommBuffer* data_;
	};

#if BFS_BACKWARD && VERTEX_SORTING
	struct BackwardReceiverProcessing : public Runnable {
		BackwardReceiverProcessing(ThisType* this__, bfs_detail::FoldCommBuffer* data__)
		: this_(this__), data_(data__) 	{ }
		virtual void run() {
			using namespace BFS_PARAMS;
			ThreadLocalBuffer* tlb = this_->thread_local_buffer_[omp_get_thread_num()];
#if VLQ_COMPRESSION
			int64_t* decode_buffer = tlb->decode_buffer;
			const uint8_t* v0_stream = data_->v0_stream->d.stream;
			const bfs_detail::PacketIndex* packet_index =
					&data_->v0_stream->d.index[data_->v0_stream->packet_index_start];
#endif
			BitmapType* restrict const visited = this_->visited_;
			BitmapType* restrict const nq_bitmap = this_->nq_bitmap_;
			BitmapType* restrict const visited_orig_bitmap = this_->visited_orig_bitmap_;
			BitmapType* restrict const nq_sorted_bitmap = this_->nq_sorted_bitmap_;
			int64_t* restrict const pred = this_->pred_;
			const int cur_level = this_->current_level_;
#if VLQ_COMPRESSION
			int v0_offset = 0, v1_offset = 0;
#endif
			int64_t num_nq_vertices = 0;

			const int log_local_verts = this_->graph_.log_local_verts();
			const int64_t log_size = get_msb_index(mpi.size_2d);
			const int64_t local_verts_mask = this_->get_number_of_local_vertices() - 1;
#define UNSWIZZLE_VERTEX(c) (((c) >> log_local_verts) | (((c) & local_verts_mask) << log_size))
#if VLQ_COMPRESSION
			for(int i = 0; i < data_->num_packets; ++i) {
				int stream_length = packet_index[i].length;
				int num_edges = packet_index[i].num_int;
#ifndef NDEBUG
				int decoded_elements =
#endif
				varint_decode_stream_signed(&v0_stream[v0_offset],
						stream_length, (uint64_t*)decode_buffer);
#if 0
				if(decoded_elements != num_edges) {
					printf("Error: Decode: decoded_elements = %d, num_edges = %d\n", decoded_elements, num_edges);
				}
#endif
				assert (decoded_elements == num_edges);
				const uint32_t* const v1_list = &data_->v1_list[v1_offset];
				int64_t v0_swizzled = 0;

				for(int c = 0; c < num_edges; ++c) {
					v0_swizzled += decode_buffer[c];
#else // VLQ_COMPRESSION
			const int64_t* const v0_list = data_->v0_list;
			const uint32_t* const v1_list = data_->v1_list;
			for(int c = 0; c < data_->num_edges; ++c) {
					int64_t v0_swizzled = v0_list[c];
#endif // VLQ_COMPRESSION
					const LocalVertsIndex v1_local = v1_list[c];
					const LocalVertsIndex word_idx = v1_local / NUMBER_PACKING_EDGE_LISTS;
					const int bit_idx = v1_local % NUMBER_PACKING_EDGE_LISTS;
					const BitmapType mask = BitmapType(1) << bit_idx;

					if((visited_orig_bitmap[word_idx] & mask) == 0) { // if this vertex has not visited
						if((__sync_fetch_and_or(&visited_orig_bitmap[word_idx], mask) & mask) == 0) {
							const int64_t v0 = UNSWIZZLE_VERTEX(v0_swizzled);
		//					const int64_t pred_v = (v0 & int64_t(0xFFFFFFFFFFFF)) | ((int64_t)cur_level << 48);
							const int64_t pred_v = v0 | (int64_t(cur_level) << 48);

							assert (pred[v1_local] == -1);
							__sync_fetch_and_or(&nq_bitmap[word_idx], mask);
							pred[v1_local] = pred_v;

							const LocalVertsIndex sorted_v1_local = this_->graph_.vertex_mapping_[v1_local];
							const LocalVertsIndex sorted_word_idx = sorted_v1_local / NUMBER_PACKING_EDGE_LISTS;
							const int sorted_bit_idx = sorted_v1_local % NUMBER_PACKING_EDGE_LISTS;
							const BitmapType sorted_mask = BitmapType(1) << sorted_bit_idx;

							__sync_fetch_and_or(&nq_sorted_bitmap[sorted_word_idx], sorted_mask);
#ifndef NDEBUG
							__sync_fetch_and_or(&visited[sorted_word_idx], sorted_mask);
#endif

							++num_nq_vertices;
						}
					}
#if VLQ_COMPRESSION
				}
				v0_offset += stream_length;
				v1_offset += num_edges;
#endif
			}
#undef UNSWIZZLE_VERTEX
			this_->comm_.relase_fold_buffer(data_);
			tlb->num_nq_vertices += num_nq_vertices;
#if 0
			this_->recv_task_.push(this);
#else
			delete this;
#endif
		}
		ThisType* const this_;
		bfs_detail::FoldCommBuffer* data_;
	};
#endif // #if BFS_BACKWARD && VERTEX_SORTING

	struct ExtractEnd : public Runnable {
		virtual void run() {
			// flush buffer
#if PROFILING_MODE
			profiling::TimeKeeper tk_all;
			profiling::TimeKeeper tk_commit;
			profiling::TimeSpan ts_commit;
#endif
			for(int i = 0; i < omp_get_num_threads(); ++i) {
				FoldPacket* packet_array =
						this_->thread_local_buffer_[i]->fold_packet;
				if(packet_array[dest_c_].num_edges > 0) {
#if PROFILING_MODE
					tk_commit.getSpanAndReset();
#endif
					this_->fold_send_packet(&packet_array[dest_c_], dest_c_);
#if PROFILING_MODE
					ts_commit += tk_commit;
#endif
					packet_array[dest_c_].num_edges = 0;
				}
			}
#if PROFILING_MODE
			tk_commit.getSpanAndReset();
#endif
			this_->comm_.fold_send_end(dest_c_);
#if PROFILING_MODE
			ts_commit += tk_commit;
			profiling::TimeSpan ts_all;
			ts_all += tk_all;
			ts_all -= ts_commit;
			this_->extract_edge_time_ += ts_all;
			this_->commit_time_ += ts_commit;
#endif
		}
		ThisType* this_;
		int dest_c_;
	};

	virtual void fold_received(bfs_detail::FoldCommBuffer* data)
	{
#if BFS_BACKWARD && VERTEX_SORTING
		if(!forward_or_backward_) {
			fiber_man_.submit(new BackwardReceiverProcessing(this, data), 1);
		} else
#endif // #if BFS_BACKWARD && VERTEX_SORTING
		{
			fiber_man_.submit(new ReceiverProcessing(this, data), 1);
		}
	}

	virtual void fold_finish()
	{
		fiber_man_.end_processing();
	}

	void printInformation()
	{
		if(mpi.isMaster() == false) return ;
		using namespace BFS_PARAMS;
		//fprintf(IMD_OUT, "Welcome to Graph500 Benchmark World.\n");
		//fprintf(IMD_OUT, "Check it out! You are running highly optimized BFS implementation.\n");

		fprintf(IMD_OUT, "===== Settings and Parameters. ====\n");
		fprintf(IMD_OUT, "NUM_BFS_ROOTS=%d.\n", NUM_BFS_ROOTS);
		fprintf(IMD_OUT, "max threads=%d.\n", omp_get_max_threads());
		fprintf(IMD_OUT, "sizeof(BitmapType)=%zd.\n", sizeof(BitmapType));
		fprintf(IMD_OUT, "Index Type of Graph: %d bytes per edge.\n", IndexArray::bytes_per_edge);
		fprintf(IMD_OUT, "sizeof(LocalVertsIndex)=%zd.\n", sizeof(LocalVertsIndex));
		fprintf(IMD_OUT, "PACKET_LENGTH=%d.\n", PACKET_LENGTH);
		fprintf(IMD_OUT, "NUM_BFS_ROOTS=%d.\n", NUM_BFS_ROOTS);
		fprintf(IMD_OUT, "NUMBER_PACKING_EDGE_LISTS=%d.\n", NUMBER_PACKING_EDGE_LISTS);
		fprintf(IMD_OUT, "NUMBER_CQ_SUMMARIZING=%d.\n", NUMBER_CQ_SUMMARIZING);
		fprintf(IMD_OUT, "MINIMUN_SIZE_OF_CQ_BITMAP=%d.\n", MINIMUN_SIZE_OF_CQ_BITMAP);
		fprintf(IMD_OUT, "BLOCK_V0_LEGNTH=%d.\n", BLOCK_V0_LEGNTH);
		fprintf(IMD_OUT, "VERVOSE_MODE=%d.\n", VERVOSE_MODE);
		fprintf(IMD_OUT, "SHARED_VISITED_OPT=%d.\n", SHARED_VISITED_OPT);
		fprintf(IMD_OUT, "VALIDATION_LEVEL=%d.\n", VALIDATION_LEVEL);
		fprintf(IMD_OUT, "DENOM_SHARED_VISITED_PART=%d.\n", DENOM_SHARED_VISITED_PART);
		fprintf(IMD_OUT, "BACKWARD_THREASOLD=%d.\n", BACKWARD_THREASOLD);
		fprintf(IMD_OUT, "BACKEARD_DENOMINATOR=%d.\n", BACKEARD_DENOMINATOR);
		fprintf(IMD_OUT, "BFS_BACKWARD=%d.\n", BFS_BACKWARD);
		fprintf(IMD_OUT, "VLQ_COMPRESSION=%d.\n", VLQ_COMPRESSION);
		fprintf(IMD_OUT, "BFS_EXPAND_COMPRESSION=%d.\n", BFS_EXPAND_COMPRESSION);
		fprintf(IMD_OUT, "VERTEX_SORTING=%d.\n", VERTEX_SORTING);
		fprintf(IMD_OUT, "BFS_FORWARD_PREFETCH=%d.\n", BFS_FORWARD_PREFETCH);
		fprintf(IMD_OUT, "BFS_BACKWARD_PREFETCH=%d.\n", BFS_BACKWARD_PREFETCH);
		fprintf(IMD_OUT, "GRAPH_BITVECTOR=%d.\n", GRAPH_BITVECTOR);
		fprintf(IMD_OUT, "GRAPH_BITVECTOR_OFFSET=%d.\n", GRAPH_BITVECTOR_OFFSET);
		fprintf(IMD_OUT, "AVOID_BUSY_WAIT=%d.\n", AVOID_BUSY_WAIT);
		fprintf(IMD_OUT, "SWITCH_FUJI_PROF=%d.\n", SWITCH_FUJI_PROF);
	}

	void prepare_sssp() { }
	void run_sssp(int64_t root, int64_t* pred) { }
	void end_sssp() { }

	// members
	FiberManager fiber_man_;
	bfs_detail::Bfs2DComm comm_;
#if 0
	ConcurrentStack<ReceiverProcessing*> recv_task_;
#endif
	ThreadLocalBuffer** thread_local_buffer_;
	BitmapType* cq_bitmap_;
	BitmapType* cq_summary_; // 128bytes -> 1bit
	BitmapType* shared_visited_;
	BitmapType* visited_;
	BitmapType* nq_bitmap_;
	BitmapType* nq_sorted_bitmap_;
	int64_t* pred_;

	struct DynamicDataSet {
		int64_t num_tmp_packets_;
		int64_t tmp_packet_offset_;
		// We count only if CQ is transfered by stream. Otherwise, 0.
		int64_t num_vertices_in_cq_;
		// This is used in backward phase.
		// The number of unvisited vertices in CQ.
		int64_t num_active_vertices_;
		int64_t num_vertices_in_nq_;
		int num_remaining_extract_jobs_;
#if AVOID_BUSY_WAIT
		pthread_mutex_t avoid_busy_wait_sync_;
#endif
#if VERVOSE_MODE
		int64_t num_edge_relax_;
#endif
	} *d_;

	int comm_length_;
	int log_local_bitmap_;
#if 0
	int num_recv_tasks_;
#endif
	int64_t tmp_packet_max_length_;
	bfs_detail::PacketIndex* tmp_packet_index_;
	ExpandCommCommand cq_comm_;
	ExpandCommCommand visited_comm_;

	struct {
		ExtractEdge* long_job;
		ExtractEdge* short_job;
		BackwardExtractEdge* back_long_job;
		BackwardExtractEdge* back_short_job;
		ExtractEnd* fold_end_job;
		int long_job_length;
		int short_job_length;
	} sched_;

	int current_level_;
	bool forward_or_backward_;

	struct {
		void* thread_local_;
	} buffer_;
#if PROFILING_MODE
	profiling::TimeSpan extract_edge_time_;
	profiling::TimeSpan vertex_scan_time_;
	profiling::TimeSpan core_proc_time_;
	profiling::TimeSpan commit_time_;
	profiling::TimeSpan recv_proc_time_;
#endif
#if BIT_SCAN_TABLE
	uint8_t bit_scan_table_[BFS_PARAMS::BIT_SCAN_TABLE_SIZE];
#endif
};

template <typename IndexArray, typename LocalVertsIndex, typename PARAMS>
void BfsBase<IndexArray, LocalVertsIndex, PARAMS>::
	run_bfs(int64_t root, int64_t* pred)
{
#if SWITCH_FUJI_PROF
	fapp_start("initialize", 0, 0);
	start_collection("initialize");
#endif
	using namespace BFS_PARAMS;
	pred_ = pred;
	cq_bitmap_ = (BitmapType*)
			page_aligned_xcalloc(sizeof(cq_bitmap_[0])*get_bitmap_size_v0());
	shared_visited_ = (BitmapType*)
			page_aligned_xcalloc(sizeof(shared_visited_[0])*get_bitmap_size_v1());
#if VERVOSE_MODE
	double tmp = MPI_Wtime();
	double start_time = tmp;
	double prev_time = tmp;
	double expand_time = 0.0, fold_time = 0.0, stall_time = 0.0;
	g_fold_send = g_fold_recv = g_bitmap_send = g_bitmap_recv = g_exs_send = g_exs_recv = 0;
#endif
	// threshold of scheduling for extracting CQ.
	const int64_t forward_sched_threshold = get_number_of_local_vertices() * mpi.size_2dr / 1024;
	const int64_t backward_sched_threshold = get_number_of_local_vertices() * mpi.size_2dr / 32;

	const int log_size = get_msb_index(mpi.size_2d);
	const int size_mask = mpi.size_2d - 1;
#define VERTEX_OWNER(v) ((v) & size_mask)
#define VERTEX_LOCAL(v) ((v) >> log_size)

	initialize_memory(pred);
#if VERVOSE_MODE
	if(mpi.isMaster()) fprintf(IMD_OUT, "Time of initialize memory: %f ms\n", (MPI_Wtime() - prev_time) * 1000.0);
	prev_time = MPI_Wtime();
	int64_t actual_traversed_edges = 0;
#endif
	current_level_ = 0;
	forward_or_backward_ = true; // begin with forward search
	int64_t global_visited_vertices = 0;
	// !!root is UNSWIZZLED.!!
	int root_owner = (int)VERTEX_OWNER(root);
	if(root_owner == mpi.rank_2d) {
		int64_t root_local = VERTEX_LOCAL(root);
		pred_[root_local] = root;
#if VERTEX_SORTING
		int64_t sortd_root_local = graph_.vertex_mapping_[root_local];
		int64_t word_idx = sortd_root_local / NUMBER_PACKING_EDGE_LISTS;
		int bit_idx = sortd_root_local % NUMBER_PACKING_EDGE_LISTS;
#else
		int64_t word_idx = root_local / NUMBER_PACKING_EDGE_LISTS;
		int bit_idx = root_local % NUMBER_PACKING_EDGE_LISTS;
#endif
		visited_[word_idx] |= BitmapType(1) << bit_idx;
		expand_root(root_local, &cq_comm_, &visited_comm_);
	}
	else {
		expand_root(-1, &cq_comm_, &visited_comm_);
	}
#if VERVOSE_MODE
	tmp = MPI_Wtime();
	if(mpi.isMaster()) fprintf(IMD_OUT, "Time of first expansion: %f ms\n", (tmp - prev_time) * 1000.0);
	expand_time += tmp - prev_time; prev_time = tmp;
#endif
#undef VERTEX_OWNER
#undef VERTEX_LOCAL

#if SWITCH_FUJI_PROF
	stop_collection("initialize");
	fapp_stop("initialize", 0, 0);
	int extract_kind;
	char *prof_mes[] = { "f_long", "f_short", "b_long", "b_short" };
#endif
	while(true) {
		++current_level_;
#if VERVOSE_MODE
		double level_start_time = MPI_Wtime();
#endif
		d_->num_vertices_in_nq_ = 0;

		fiber_man_.begin_processing();
		comm_.begin_fold_comm(forward_or_backward_);

#if 1 // improved scheduling
		// submit graph extraction job
		if(forward_or_backward_) {
			// forward search
			if(d_->num_vertices_in_cq_ >= forward_sched_threshold) {
				fiber_man_.submit_array(sched_.long_job, sched_.long_job_length, 0);
				d_->num_remaining_extract_jobs_ = sched_.long_job_length;
#if SWITCH_FUJI_PROF
				extract_kind = 0;
#endif
			}
			else {
				fiber_man_.submit_array(sched_.short_job, sched_.short_job_length, 0);
				d_->num_remaining_extract_jobs_ = sched_.short_job_length;
#if SWITCH_FUJI_PROF
				extract_kind = 1;
#endif
			}
		}
		else {
			// backward search
			if(d_->num_active_vertices_ >= backward_sched_threshold) {
				fiber_man_.submit_array(sched_.back_long_job, sched_.long_job_length, 0);
				d_->num_remaining_extract_jobs_ = sched_.long_job_length;
#if SWITCH_FUJI_PROF
				extract_kind = 2;
#endif
			}
			else {
				fiber_man_.submit_array(sched_.back_short_job, sched_.short_job_length, 0);
				d_->num_remaining_extract_jobs_ = sched_.short_job_length;
#if SWITCH_FUJI_PROF
				extract_kind = 3;
#endif
			}
		}
#else
		// submit graph extraction job
		if(d_->num_vertices_in_cq_ >= sched_threshold) {
			if(forward_or_backward_)
			fiber_man_.submit_array(sched_.long_job, sched_.long_job_length, 0);
			else
				fiber_man_.submit_array(sched_.back_long_job, sched_.long_job_length, 0);

			d_->num_remaining_extract_jobs_ = sched_.long_job_length;
		}
		else {
			if(forward_or_backward_)
			fiber_man_.submit_array(sched_.short_job, sched_.short_job_length, 0);
			else
				fiber_man_.submit_array(sched_.back_short_job, sched_.short_job_length, 0);

			d_->num_remaining_extract_jobs_ = sched_.short_job_length;
		}
#endif
#if SWITCH_FUJI_PROF
		fapp_start(prof_mes[extract_kind], 0, 0);
		start_collection(prof_mes[extract_kind]);
#endif
#if VERVOSE_MODE
		if(mpi.isMaster()) {
			double nq_percent = (double)d_->num_vertices_in_cq_ / (get_actual_number_of_local_vertices() * mpi.size_2dr) * 100.0;
			double visited_percent = (double)global_visited_vertices / get_actual_number_of_global_vertices() * 100.0;
			fprintf(IMD_OUT, "Level %d (%s) start, # of tasks : %d\n", current_level_,
					forward_or_backward_ ? "Forward" : "Backward", d_->num_remaining_extract_jobs_);
			fprintf(IMD_OUT, "NQ: %f %%, Visited: %f %%", nq_percent, visited_percent);
			if(forward_or_backward_)
				fprintf(IMD_OUT, "\n");
			else
				fprintf(IMD_OUT, ", Active: %f %%\n",
						(double)d_->num_active_vertices_ / (get_actual_number_of_local_vertices() * mpi.size_2dc) * 100.0);
		}
		d_->num_edge_relax_ = 0;
#endif
		d_->num_active_vertices_ = 0;
#if PROFILING_MODE
		fiber_man_.reset_wait_time();
#endif

#pragma omp parallel
		fiber_man_.enter_processing();

		assert(comm_.check_num_send_buffer());
#if VERVOSE_MODE
		tmp = MPI_Wtime(); fold_time += tmp - prev_time; prev_time = tmp;
#endif
#if SWITCH_FUJI_PROF
		stop_collection(prof_mes[extract_kind]);
		fapp_stop(prof_mes[extract_kind], 0, 0);
		fapp_start("sync", 0, 0);
		start_collection("sync");
#endif
		int64_t num_nq_vertices = d_->num_vertices_in_nq_;

		int64_t global_nq_vertices;
		MPI_Allreduce(&num_nq_vertices, &global_nq_vertices, 1,
				get_mpi_type(num_nq_vertices), MPI_SUM, mpi.comm_2d);
		global_visited_vertices += global_nq_vertices;
#if VERVOSE_MODE
		tmp = MPI_Wtime(); stall_time += tmp - prev_time; prev_time = tmp;
		actual_traversed_edges += d_->num_edge_relax_;
#endif
#if PROFILING_MODE
		if(forward_or_backward_)
			extract_edge_time_.submit("forward edge", current_level_);
		else
			extract_edge_time_.submit("backward edge", current_level_);

		commit_time_.submit("extract commit", current_level_);
		vertex_scan_time_.submit("vertex scan time", current_level_);
#if DETAILED_PROF_MODE
		core_proc_time_.submit("core processing time", current_level_);
#endif
		recv_proc_time_.submit("recv proc", current_level_);
		comm_.submit_prof_info("send comm wait", current_level_);
		fiber_man_.submit_wait_time("fiber man wait", current_level_);
		profiling::g_pis.submitCounter(d_->num_edge_relax_, "edge relax", current_level_);
#endif
#if DEBUG_PRINT
	if(mpi.isMaster()) printf("global_nq_vertices=%"PRId64"\n", global_nq_vertices);
#endif
#if SWITCH_FUJI_PROF
		stop_collection("sync");
		fapp_stop("sync", 0, 0);
#endif
		if(global_nq_vertices == 0)
			break;
#if SWITCH_FUJI_PROF
		fapp_start("expand", 0, 0);
		start_collection("expand");
#endif
		expand(global_nq_vertices, global_visited_vertices, &cq_comm_, &visited_comm_);
#if SWITCH_FUJI_PROF
		stop_collection("expand");
		fapp_stop("expand", 0, 0);
#endif
#if VERVOSE_MODE
		tmp = MPI_Wtime();
		double time_of_level = MPI_Wtime() - level_start_time;
		if(mpi.isMaster()) {
			fprintf(IMD_OUT, "Edge relax: %f M/s (%"PRId64"), Time: %f ms\n",
					d_->num_edge_relax_ / 1000000.0 / time_of_level, d_->num_edge_relax_, time_of_level * 1000.0);
		}
		expand_time += tmp - prev_time; prev_time = tmp;
#endif
	}
#if PROFILING_MODE
	comm_.submit_mem_info();
#endif
#if VERVOSE_MODE
	if(mpi.isMaster()) fprintf(IMD_OUT, "Time of BFS: %f ms\n", (MPI_Wtime() - start_time) * 1000.0);
	double time3[3] = { fold_time, expand_time, stall_time };
	double timesum3[3];
	int64_t commd[7] = { g_fold_send, g_fold_recv, g_bitmap_send, g_bitmap_recv, g_exs_send, g_exs_recv, actual_traversed_edges };
	int64_t commdsum[7];
	MPI_Reduce(time3, timesum3, 3, MPI_DOUBLE, MPI_SUM, 0, mpi.comm_2d);
	MPI_Reduce(commd, commdsum, 7, get_mpi_type(commd[0]), MPI_SUM, 0, mpi.comm_2d);
	if(mpi.isMaster()) {
		fprintf(IMD_OUT, "Avg time of fold: %f ms\n", timesum3[0] / mpi.size_2d * 1000.0);
		fprintf(IMD_OUT, "Avg time of expand: %f ms\n", timesum3[1] / mpi.size_2d * 1000.0);
		fprintf(IMD_OUT, "Avg time of stall: %f ms\n", timesum3[2] / mpi.size_2d * 1000.0);
		fprintf(IMD_OUT, "Avg fold_send: %"PRId64"\n", commdsum[0] / mpi.size_2d);
		fprintf(IMD_OUT, "Avg fold_recv: %"PRId64"\n", commdsum[1] / mpi.size_2d);
		fprintf(IMD_OUT, "Avg bitmap_send: %"PRId64"\n", commdsum[2] / mpi.size_2d);
		fprintf(IMD_OUT, "Avg bitmap_recv: %"PRId64"\n", commdsum[3] / mpi.size_2d);
		fprintf(IMD_OUT, "Avg exs_send: %"PRId64"\n", commdsum[4] / mpi.size_2d);
		fprintf(IMD_OUT, "Avg exs_recv: %"PRId64"\n", commdsum[5] / mpi.size_2d);
		fprintf(IMD_OUT, "Actual traversed edges: %"PRId64" (%f %%)\n", commdsum[6],
				(double)commdsum[6] / graph_.num_global_edges_ * 100.0);
	}
#endif
	free(cq_bitmap_); cq_bitmap_ = NULL;
	free(shared_visited_); shared_visited_ = NULL;
}

#endif /* BFS_HPP_ */
