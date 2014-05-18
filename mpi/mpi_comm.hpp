/*
 * mpi_comm.hpp
 *
 *  Created on: 2014/05/17
 *      Author: ueno
 */

#ifndef MPI_COMM_HPP_
#define MPI_COMM_HPP_

#include "abstract_comm.hpp"

class MpiAlltoallCommunicatorBase : public AlltoallCommunicator {
	struct CommTarget {
		CommTarget()
			: send_buf(NULL)
			, recv_buf(NULL) {
		}

		std::deque<CommunicationBuffer*> send_queue;
		CommunicationBuffer* send_buf;
		CommunicationBuffer* recv_buf;
	};
public:
	MpiAlltoallCommunicatorBase() {
		node_list_length_ = 0;
		node_ = NULL;
		mpi_reqs_ = NULL;
		num_pending_send = 0;
	}
	virtual ~MpiAlltoallCommunicatorBase() {
		delete [] node_;
		delete [] mpi_reqs_;
	}
	virtual void send(CommunicationBuffer* data, int target) {
		node_[target].send_queue.push_back(data);
		++num_pending_send;
		set_send_buffer(target);
	}
	virtual AlltoallSubCommunicator reg_comm(AlltoallCommParameter parm) {
		int idx = handlers_.size();
		handlers_.push_back(parm);
		int comm_size;
		MPI_Comm_size(parm.base_communicator, &comm_size);
		node_list_length_ = std::max(comm_size, node_list_length_);
		return idx;
	}
	virtual AlltoallBufferHandler* begin(AlltoallSubCommunicator sub_comm) {
		AlltoallCommParameter active = handlers_[sub_comm];
		comm_ = active.base_communicator;
		tag_ = active.tag;
		handler_ = active.handler;
		data_type_ = handler_->data_type();
		MPI_Comm_size(comm_, &comm_size_);
		initialized_ = false;

		if(node_ == NULL) {
			node_ = new CommTarget[node_list_length_]();
			mpi_reqs_ = new MPI_Request[node_list_length_*REQ_TOTAL]();
			for(int i = 0; i < node_list_length_; ++i) {
				for(int k = 0; k < REQ_TOTAL; ++k) {
					mpi_reqs_[REQ_TOTAL*i + k] = MPI_REQUEST_NULL;
				}
			}
		}

		return handler_;
	}
	//! @return finished
	virtual bool probe() {
		if(initialized_ == false) {
			initialized_ = true;

			num_recv_active = num_send_active = comm_size_;
			for(int i = 0; i < comm_size_; ++i) {
				CommTarget& node = node_[i];
				assert (node.recv_buf == NULL);
				node.recv_buf = handler_->alloc_buffer();
				assert (node.recv_buf != NULL);
				set_recv_buffer(node.recv_buf, i, &mpi_reqs_[REQ_TOTAL*i + REQ_RECV]);
			}
		}

		if(num_recv_active == 0 && num_send_active == 0) {
			// finished
			handler_->finished();
			return true;
		}

		int index;
		int flag;
		MPI_Status status;
		MPI_Testany(comm_size_ * (int)REQ_TOTAL, mpi_reqs_, &index, &flag, &status);

		if(flag != 0 && index != MPI_UNDEFINED) {
			const int src_c = index/REQ_TOTAL;
			const MPI_REQ_INDEX req_kind = (MPI_REQ_INDEX)(index%REQ_TOTAL);
			const bool b_send = (req_kind == REQ_SEND);

			CommTarget& node = node_[src_c];
			CommunicationBuffer* buf;
			if(b_send) {
				buf = node.send_buf; node.send_buf = NULL;
			}
			else {
				buf = node.recv_buf; node.recv_buf = NULL;
			}

			assert (mpi_reqs_[index] == MPI_REQUEST_NULL);
			mpi_reqs_[index] = MPI_REQUEST_NULL;

			if(req_kind == REQ_RECV) {
				MPI_Get_count(&status, data_type_, &buf->length_);
			}

			bool completion_message = (buf->length_ == 0);
			// complete
			if(b_send) {
				// send buffer
				handler_->free_buffer(buf);
				if(completion_message) {
					// sent fold completion
					--num_send_active;
				}
				else {
					set_send_buffer(src_c);
				}
			}
			else {
				// recv buffer
				if(completion_message) {
					// received fold completion
					--num_recv_active;
					handler_->free_buffer(buf);
				}
				else {
					// set new buffer for next receiving
					recv_stv.push_back(src_c);

					handler_->received(buf, src_c);
				}
			}

			// process recv starves
			while(recv_stv.size() > 0) {
				int target = recv_stv.front();
				CommTarget& node = node_[target];
				assert (node.recv_buf == NULL);
				node.recv_buf = handler_->alloc_buffer();
				if(node.recv_buf == NULL) break;
				set_recv_buffer(node.recv_buf, target, &mpi_reqs_[REQ_TOTAL*target + REQ_RECV]);
				recv_stv.pop_front();
			}
		}

		return false;
	}
	virtual int get_comm_size() {
		return comm_size_;
	}
#ifndef NDEBUG
	bool check_num_send_buffer() { return (num_pending_send == 0); }
#endif

private:

	enum MPI_REQ_INDEX {
		REQ_SEND = 0,
		REQ_RECV = 1,
		REQ_TOTAL = 2,
	};

	std::vector<AlltoallCommParameter> handlers_;
	MPI_Comm comm_;
	int tag_;
	AlltoallBufferHandler* handler_;
	MPI_Datatype data_type_;
	int comm_size_;
	bool initialized_;

	int node_list_length_;
	CommTarget* node_;
	std::deque<int> recv_stv;
	MPI_Request* mpi_reqs_;

	int num_recv_active;
	int num_send_active;
	int num_pending_send;

	void set_send_buffer(int target) {
		CommTarget& node = node_[target];
		if(node.send_buf) {
			// already sending
			return ;
		}
		if(node.send_queue.size() > 0) {
			CommunicationBuffer* buf = node.send_buf = node.send_queue.front();
			node.send_queue.pop_front();
			MPI_Request* req = &mpi_reqs_[REQ_TOTAL*target + REQ_SEND];

			MPI_Isend(buf->pointer(), buf->length_, data_type_,
					target, tag_, comm_, req);

			--num_pending_send;
		}
	}

	void set_recv_buffer(CommunicationBuffer* buf, int target, MPI_Request* req) {
		MPI_Irecv(buf->pointer(), handler_->buffer_length(), data_type_,
				target, tag_, comm_, req);
	}

};

template <typename T>
class MpiAlltoallCommunicator : public MpiAlltoallCommunicatorBase {
public:
	MpiAlltoallCommunicator() : MpiAlltoallCommunicatorBase() { }
	memory::Pool<T>* get_allocator() {
		return &pool_;
	}
#if PROFILING_MODE
	void submit_prof_info(int number) {
		profiling::g_pis.submitCounter(pool_.num_extra_buffer_, "num_extra_buffer_", number);
	}
#endif
private:

	class CommBufferPool : public memory::ConcurrentPool<T> {
	public:
		CommBufferPool()
			: memory::ConcurrentPool<T>()
		{
			num_extra_buffer_ = 0;
		}

		int num_extra_buffer_;
	protected:
		virtual T* allocate_new() {
			PROF(__sync_fetch_and_add(&num_extra_buffer_, 1));
			return new (page_aligned_xmalloc(sizeof(T))) T();
		}
	};

	CommBufferPool pool_;
};

#endif /* MPI_COMM_HPP_ */
