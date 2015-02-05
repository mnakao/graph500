/*
 * abstract_comm.hpp
 *
 *  Created on: 2014/05/17
 *      Author: ueno
 */

#ifndef ABSTRACT_COMM_HPP_
#define ABSTRACT_COMM_HPP_

#include <limits.h>
#include "utils.hpp"
#include "fiber.hpp"

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
};

class AsyncAlltoallManager : public Runnable {
	struct Buffer {
		void* ptr;
		int length;
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
	};
public:
	AsyncAlltoallManager(MPI_Comm comm_, AlltoallBufferHandler* buffer_provider_)
		: buffer_provider_(buffer_provider_)
		, scatter_(comm_)
	{
		CTRACER(AsyncA2A_construtor);
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
		CTRACER(prepare);
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
		CTRACER(comm_send);
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

	void run() {
		// merge
		PROF(profiling::TimeKeeper tk_all);
		int es = buffer_provider_->element_size();
		USER_START(a2a_merge);
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
				for(int b = 0; b < (int)node.send_data.size(); ++b) {
					void* ptr = node.send_data[b].ptr;
					int offset = offsets[i];
					int length = node.send_data[b].length;
					memcpy(dst + offset * es, ptr, length * es);
					offsets[i] += length;
				}
				node.send_data.clear();
			} // #pragma omp for schedule(static)
		} // #pragma omp parallel
		USER_END(a2a_merge);

		void* sendbuf = buffer_provider_->second_buffer();
		void* recvbuf = buffer_provider_->clear_buffers();
		MPI_Datatype type = buffer_provider_->data_type();
		int recvbufsize = buffer_provider_->max_size();
		PROF(merge_time_ += tk_all);
		USER_START(a2a_comm);
		scatter_.alltoallv(sendbuf, recvbuf, type, recvbufsize);
		PROF(comm_time_ += tk_all);
		USER_END(a2a_comm);

		VERVOSE(last_send_size_ = scatter_.get_send_count() * es);
		VERVOSE(last_recv_size_ = scatter_.get_recv_count() * es);

		int* recv_offsets = scatter_.get_recv_offsets();

#pragma omp parallel for schedule(dynamic,1)
		for(int i = 0; i < comm_size_; ++i) {
			int offset = recv_offsets[i];
			int length = recv_offsets[i+1] - offset;
			buffer_provider_->received(recvbuf, offset, length, i);
		}
		PROF(recv_proc_time_ += tk_all);
	}
#if PROFILING_MODE
	void submit_prof_info(int level) {
		merge_time_.submit("merge a2a data", level);
		comm_time_.submit("a2a comm", level);
		recv_proc_time_.submit("proc recv data", level);
		profiling::g_pis.submitCounter(last_send_size_, "a2a send data", level);
		profiling::g_pis.submitCounter(last_recv_size_, "a2a recv data", level);
	}
#endif
#if VERVOSE_MODE
	int get_last_send_size() { return last_send_size_; }
#endif
private:

	struct DynamicDataSet {
		// lock topology
		// FoldNode::send_mutex -> thread_sync_
		pthread_mutex_t thread_sync_;
	} *d_;

	int buffer_size_;
	int comm_size_;

	int node_list_length_;
	CommTarget* node_;
	AlltoallBufferHandler* buffer_provider_;
	ScatterContext scatter_;

	PROF(profiling::TimeSpan merge_time_);
	PROF(profiling::TimeSpan comm_time_);
	PROF(profiling::TimeSpan recv_proc_time_);
	VERVOSE(int last_send_size_);
	VERVOSE(int last_recv_size_);

	void flush(CommTarget& node) {
		if(node.cur_buf.ptr != NULL) {
			node.cur_buf.length = node.filled_size_;
			node.send_data.push_back(node.cur_buf);
			node.cur_buf.ptr = NULL;
		}
	}

	void* get_send_buffer() {
		CTRACER(get_send_buffer);
		pthread_mutex_lock(&d_->thread_sync_);
		void* ret = buffer_provider_->get_buffer();
		pthread_mutex_unlock(&d_->thread_sync_);
		return ret;
	}
};
#undef debug

#endif /* ABSTRACT_COMM_HPP_ */

