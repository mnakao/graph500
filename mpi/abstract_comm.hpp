#ifndef ABSTRACT_COMM_HPP_
#define ABSTRACT_COMM_HPP_

#include <limits.h>
#include "utils.hpp"

#define debug(...) debug_print(ABSCO, __VA_ARGS__)
class AlltoallBufferHandler {
public:
	virtual ~AlltoallBufferHandler() { }
	virtual void* get_buffer() = 0;
	virtual void add(void* buffer, void* data, int offset, int length) = 0;
	virtual void* clear_buffers() = 0;
	virtual void* second_buffer() = 0;
	virtual int max_size() = 0;
	virtual int buffer_length() = 0;
	virtual MPI_Datatype data_type() = 0;
	virtual int element_size() = 0;
	virtual void received(void* buf, int offset, int length, int from) = 0;
	virtual void finish() = 0;
};

class AsyncAlltoallManager {
	struct Buffer {
		void* ptr;
		int length;
	};

	struct PointerData {
		void* ptr;
		int length;
		int64_t header;
	};

	struct CommTarget {
		CommTarget()
			: reserved_size_(0)
			, filled_size_(0) {
			cur_buf.ptr = NULL;
			cur_buf.length = 0;
			pthread_mutex_init(&send_mutex, NULL);
		}
		~CommTarget() {
			pthread_mutex_destroy(&send_mutex);
		}

		pthread_mutex_t send_mutex;
		// monitor : send_mutex
		volatile int reserved_size_;
		volatile int filled_size_;
		Buffer cur_buf;
		std::vector<Buffer> send_data;
		std::vector<PointerData> send_ptr;
	};
public:
	AsyncAlltoallManager(MPI_Comm comm_, AlltoallBufferHandler* buffer_provider_)
		: comm_(comm_)
		, buffer_provider_(buffer_provider_)
		, scatter_(comm_)
	{
		MPI_Comm_size(comm_, &comm_size_);
		node_ = new CommTarget[comm_size_]();
		d_ = new DynamicDataSet();
		pthread_mutex_init(&d_->thread_sync_, NULL);
		buffer_size_ = buffer_provider_->buffer_length();
	}
	virtual ~AsyncAlltoallManager() {
		delete [] node_; node_ = NULL;
	}

	void prepare() {
		debug("prepare idx=%d", sub_comm);
		for(int i = 0; i < comm_size_; ++i) {
			node_[i].reserved_size_ = node_[i].filled_size_ = buffer_size_;
		}
	}

	/**
	 * Asynchronous send.
	 * When the communicator receive data, it will call fold_received(FoldCommBuffer*) function.
	 * To reduce the memory consumption, when the communicator detects stacked jobs,
	 * it also process the tasks in the fiber_man_ except the tasks that have the lowest priority (0).
	 * This feature realize the fixed memory consumption.
	 */
	void put(void* ptr, int length, int target)
	{
		if(length == 0) {
			assert(length > 0);
			return ;
		}
		CommTarget& node = node_[target];

//#if ASYNC_COMM_LOCK_FREE
		do {
			int offset = __sync_fetch_and_add(&node.reserved_size_, length);
			if(offset > buffer_size_) {
				// wait
				while(node.reserved_size_ > buffer_size_) ;
				continue ;
			}
			else if(offset + length > buffer_size_) {
				// swap buffer
				assert (offset > 0);
				while(offset != node.filled_size_) ;
				flush(node);
				node.cur_buf.ptr = get_send_buffer(); // Maybe, this takes much time.
				// This order is important.
				offset = node.filled_size_ = 0;
				__sync_synchronize(); // membar
				node.reserved_size_ = length;
			}
			buffer_provider_->add(node.cur_buf.ptr, ptr, offset, length);
			__sync_fetch_and_add(&node.filled_size_, length);
			break;
		} while(true);
// #endif
	}

	void put_ptr(void* ptr, int length, int64_t header, int target) {
		CommTarget& node = node_[target];
		PointerData data = { ptr, length, header };

		pthread_mutex_lock(&node.send_mutex);
		node.send_ptr.push_back(data);
		pthread_mutex_unlock(&node.send_mutex);
	}

	void run_with_ptr() {
		int es = buffer_provider_->element_size();
		int max_size = buffer_provider_->max_size() / (es * comm_size_);

		const int MINIMUM_POINTER_SPACE = 40;

		for(int loop = 0; ; ++loop) {
#pragma omp parallel
			{
				int* counts = scatter_.get_counts();
#pragma omp for schedule(static)
				for(int i = 0; i < comm_size_; ++i) {
					CommTarget& node = node_[i];
					flush(node);
					for(int b = 0; b < (int)node.send_data.size(); ++b) {
						counts[i] += node.send_data[b].length;
					}
					for(int b = 0; b < (int)node.send_ptr.size(); ++b) {
						PointerData& buffer = node.send_ptr[b];
						int length = buffer.length;
						if(length == 0) continue;

						int size = length + 3;
						if(counts[i] + size >= max_size) {
							counts[i] = max_size;
							break;
						}

						counts[i] += size;
						if(counts[i] + MINIMUM_POINTER_SPACE >= max_size) {
							// too small space
							break;
						}
					}
				} // #pragma omp for schedule(static)
			}

			scatter_.sum();

			if(loop > 0) {
				int has_data = (scatter_.get_send_count() > 0);
				MPI_Allreduce(MPI_IN_PLACE, &has_data, 1, MPI_INT, MPI_LOR, comm_);
				if(has_data == 0) break;
			}

#pragma omp parallel
			{
				int* offsets = scatter_.get_offsets();
				uint8_t* dst = (uint8_t*)buffer_provider_->second_buffer();
#pragma omp for schedule(static)
				for(int i = 0; i < comm_size_; ++i) {
					CommTarget& node = node_[i];
					int& offset = offsets[i];
					int count = 0;
					for(int b = 0; b < (int)node.send_data.size(); ++b) {
						Buffer buffer = node.send_data[b];
						void* ptr = buffer.ptr;
						int length = buffer.length;
						memcpy(dst + offset * es, ptr, length * es);
						offset += length;
						count += length;
					}
					for(int b = 0; b < (int)node.send_ptr.size(); ++b) {
						PointerData& buffer = node.send_ptr[b];
						int64_t* ptr = (int64_t*)buffer.ptr;
						int length = buffer.length;
						if(length == 0) continue;

						int size = length + 3;
						if(count + size >= max_size) {
							length = max_size - count - 3;
							count = max_size;
						}
						else {
							count += size;
						}
						uint32_t* dst_ptr = (uint32_t*)&dst[offset * es];
						dst_ptr[0] = (buffer.header >> 32) | 0x80000000u | 0x40000000u;
						dst_ptr[1] = (uint32_t)buffer.header;
						dst_ptr[2] = length;
						dst_ptr += 3;
						for(int i = 0; i < length; ++i) {
							dst_ptr[i] = ptr[i] & 0x7FFFFFFF;
						}
						offset += 3 + length;

						buffer.length -= length;
						buffer.ptr = (int64_t*)buffer.ptr + length;

						if(count + MINIMUM_POINTER_SPACE >= max_size) break;
					}
					node.send_data.clear();
				} // #pragma omp for schedule(static)
			} // #pragma omp parallel

			void* sendbuf = buffer_provider_->second_buffer();
			void* recvbuf = buffer_provider_->clear_buffers();
			MPI_Datatype type = buffer_provider_->data_type();
			int recvbufsize = buffer_provider_->max_size();
			scatter_.alltoallv(sendbuf, recvbuf, type, recvbufsize);
			int* recv_offsets = scatter_.get_recv_offsets();

#pragma omp parallel for
			for(int i = 0; i < comm_size_; ++i) {
				int offset = recv_offsets[i];
				int length = recv_offsets[i+1] - offset;
				buffer_provider_->received(recvbuf, offset, length, i);
			}
			buffer_provider_->finish();
		}

		// clear
		for(int i = 0; i < comm_size_; ++i) {
			CommTarget& node = node_[i];
			node.send_ptr.clear();
		}

	}

	void run() {
		// merge
		int es = buffer_provider_->element_size();
#pragma omp parallel
		{
			int* counts = scatter_.get_counts();
#pragma omp for schedule(static)
			for(int i = 0; i < comm_size_; ++i) {
				CommTarget& node = node_[i];
				flush(node);
				for(int b = 0; b < (int)node.send_data.size(); ++b) {
					counts[i] += node.send_data[b].length;
				}
			} // #pragma omp for schedule(static)
		}

		scatter_.sum();

#pragma omp parallel
		{
			int* offsets = scatter_.get_offsets();
			uint8_t* dst = (uint8_t*)buffer_provider_->second_buffer();
#pragma omp for schedule(static)
			for(int i = 0; i < comm_size_; ++i) {
				CommTarget& node = node_[i];
				int& offset = offsets[i];
				for(int b = 0; b < (int)node.send_data.size(); ++b) {
					Buffer buffer = node.send_data[b];
					void* ptr = buffer.ptr;
					int length = buffer.length;
					memcpy(dst + offset * es, ptr, length * es);
					offset += length;
				}
				node.send_data.clear();
			} // #pragma omp for schedule(static)
		} // #pragma omp parallel

		void* sendbuf = buffer_provider_->second_buffer();
		void* recvbuf = buffer_provider_->clear_buffers();
		MPI_Datatype type = buffer_provider_->data_type();
		int recvbufsize = buffer_provider_->max_size();
		scatter_.alltoallv(sendbuf, recvbuf, type, recvbufsize);
		int* recv_offsets = scatter_.get_recv_offsets();

#pragma omp parallel for schedule(dynamic,1)
		for(int i = 0; i < comm_size_; ++i) {
			int offset = recv_offsets[i];
			int length = recv_offsets[i+1] - offset;
			buffer_provider_->received(recvbuf, offset, length, i);
		}
	}
private:

	struct DynamicDataSet {
		// lock topology
		// FoldNode::send_mutex -> thread_sync_
		pthread_mutex_t thread_sync_;
	} *d_;

	MPI_Comm comm_;
	int buffer_size_;
	int comm_size_;
	int node_list_length_;
	CommTarget* node_;
	AlltoallBufferHandler* buffer_provider_;
	ScatterContext scatter_;

	void flush(CommTarget& node) {
		if(node.cur_buf.ptr != NULL) {
			node.cur_buf.length = node.filled_size_;
			node.send_data.push_back(node.cur_buf);
			node.cur_buf.ptr = NULL;
		}
	}

	void* get_send_buffer() {
		pthread_mutex_lock(&d_->thread_sync_);
		void* ret = buffer_provider_->get_buffer();
		pthread_mutex_unlock(&d_->thread_sync_);
		return ret;
	}
};

#undef debug

#endif /* ABSTRACT_COMM_HPP_ */
