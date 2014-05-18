/*
 * abstract_comm.hpp
 *
 *  Created on: 2014/05/17
 *      Author: ueno
 */

#ifndef ABSTRACT_COMM_HPP_
#define ABSTRACT_COMM_HPP_

#include "utils.hpp"
#include "fiber.hpp"

class CommunicationBuffer {
public:
	virtual ~CommunicationBuffer() { }
	virtual void add(void* data, int offset, int length) = 0;
	virtual void* base_object() = 0;
	virtual int bytes() = 0;
	virtual void* pointer() = 0;
	int length_;
};

class CommCommand {
public:
	virtual ~CommCommand() { }
	virtual void comm_cmd() = 0;
};

class AlltoallBufferHandler {
public:
	virtual ~AlltoallBufferHandler() { }
	virtual void received(CommunicationBuffer* buf, int src) = 0;
	virtual void finished() = 0;
	virtual CommunicationBuffer* alloc_buffer() = 0;
	virtual void free_buffer(CommunicationBuffer*) = 0;
	virtual int buffer_length() = 0;
	virtual MPI_Datatype data_type() = 0;
};

struct AlltoallCommParameter {
	MPI_Comm base_communicator;
	int tag;
	int num_nics_to_use;
	AlltoallBufferHandler* handler;

	AlltoallCommParameter(MPI_Comm comm__,
			int tag__,
			AlltoallBufferHandler* handler__) {
		base_communicator = comm__;
		tag = tag__;
		handler = handler__;
	}
};

typedef int AlltoallSubCommunicator;

class AlltoallCommunicator {
public:
	virtual ~AlltoallCommunicator() { }
	virtual void send(CommunicationBuffer* data, int target) = 0;
	virtual AlltoallSubCommunicator reg_comm(AlltoallCommParameter parm) = 0;
	virtual AlltoallBufferHandler* begin(AlltoallSubCommunicator sub_comm) = 0;
	//! @return finished
	virtual bool probe() = 0;
	virtual int get_comm_size() = 0;
};

class AsyncCommHandler {
public:
	virtual ~AsyncCommHandler() { }
	virtual void probe() = 0;
};

class AsyncAlltoallManager : public Runnable {
	struct CommTarget {
		CommTarget()
			: reserved_size_(0)
			, filled_size_(0)
			, cur_buf(NULL) {
			pthread_mutex_init(&send_mutex, NULL);
		}
		~CommTarget() {
			pthread_mutex_destroy(&send_mutex);
		}

		pthread_mutex_t send_mutex;
		// monitor : send_mutex
		volatile int reserved_size_;
		volatile int filled_size_;
		CommunicationBuffer* cur_buf;
	};
public:
	AsyncAlltoallManager(AlltoallCommunicator* comm__, FiberManager* fiber_man__) {
		comm_ = comm__;
		fiber_man_ = fiber_man__;
		comm_size_ = 0;
		node_list_length_ = 0;
		node_ = NULL;
		send_queue_limit_ = INT_MAX;

		d_ = new DynamicDataSet();
		pthread_mutex_init(&d_->thread_sync_, NULL);
		d_->command_active_ = false;
	}

	void prepare(AlltoallSubCommunicator sub_comm) {
		buffer_provider_ = comm_->begin(sub_comm);
		comm_size_ = comm_->get_comm_size();
		if(node_list_length_ < comm_size_) {
			delete [] node_;
			node_ = new CommTarget[comm_size_]();
			node_list_length_ = comm_size_;
		}
		buffer_size_ = buffer_provider_->buffer_length();
		for(int i = 0; i < comm_size_; ++i) {
			node_[i].reserved_size_ = node_[i].filled_size_ = buffer_size_;
		}
		alltoall_finished_ = false;
	}

	/**
	 * Asynchronous send.
	 * When the communicator receive data, it will call fold_received(FoldCommBuffer*) function.
	 * To reduce the memory consumption, when the communicator detects stacked jobs,
	 * it also process the tasks in the fiber_man_ except the tasks that have the lowest priority (0).
	 * This feature realize the fixed memory consumption.
	 */
	template <bool proc>
	void send(void* ptr, int length, int target)
	{
		VT_TRACER("comm_send");
		if(length == 0) {
			assert(length > 0);
			return ;
		}
		CommTarget& node = node_[target];
		bool process_task = false;

//#if ASYNC_COMM_LOCK_FREE
		do {
			int offset = __sync_fetch_and_add(&node.reserved_size_, length);
			if(offset > buffer_size_) {
				// wait
				int count = 0;
				while(node.reserved_size_ > buffer_size_) {
					if(count++ >= 1000) {
						if(proc) {
							if(fiber_man_->process_task(1)) while(fiber_man_->process_task(1)); // process recv task
							else sched_yield();
						}
						//else sched_yield();
					}
				}
				continue ;
			}
			else if(offset + length > buffer_size_) {
				// swap buffer
				assert (offset > 0);
				while(offset != node.filled_size_) ;
				if(node.cur_buf != NULL) {
					send_submit(target);
				}
				node.cur_buf = get_send_buffer<proc>(); // Maybe, this takes much time.
				// This order is important.
				offset = node.filled_size_ = 0;
				__sync_synchronize(); // membar
				node.reserved_size_ = length;
				process_task = true;
			}
			node.cur_buf->add(ptr, offset, length);
			__sync_fetch_and_add(&node.filled_size_, length);
			break;
		} while(true);

		if(proc && process_task) {
			while(fiber_man_->process_task(1)); // process recv task
		}
// #endif
	}

	void send_end(int target)
	{
		CommTarget& node = node_[target];
		assert (node.reserved_size_ == node.filled_size_);

		if(node.filled_size_ > 0 && node.cur_buf != NULL) {
			send_submit(target);
		}
		if(node.cur_buf == NULL) {
			node.cur_buf = get_send_buffer<false>();
		}

		node.reserved_size_ = node.filled_size_ = 0;
		send_submit(target);
		assert(node.cur_buf == NULL);
	}

	void input_command(CommCommand* comm)
	{
		InternalCommand cmd;
		cmd.kind = MANUAL_CMD;
		cmd.cmd = comm;
		put_command(cmd);
	}

	void register_handler(AsyncCommHandler* comm)
	{
		InternalCommand cmd;
		cmd.kind = ADD_HANDLER;
		cmd.handler = comm;
		put_command(cmd);
	}

	void remove_handler(AsyncCommHandler* comm)
	{
		InternalCommand cmd;
		cmd.kind = REMOVE_HANDLER;
		cmd.handler = comm;
		put_command(cmd);
	}

	virtual void run() {
		VT_TRACER("comm_routine");

		// command loop
		while(true) {
			if(d_->command_active_) {
				pthread_mutex_lock(&d_->thread_sync_);
				InternalCommand cmd;
				while(pop_command(&cmd)) {
					pthread_mutex_unlock(&d_->thread_sync_);
					switch(cmd.kind) {
					case SEND:
						comm_->send(cmd.send.data, cmd.send.target);
						break;
					case MANUAL_CMD:
						cmd.cmd->comm_cmd();
						break;
					case ADD_HANDLER:
						async_comm_handlers_.push_back(cmd.handler);
						break;
					case REMOVE_HANDLER:
						for(std::vector<AsyncCommHandler*>::iterator it = async_comm_handlers_.begin();
								it != async_comm_handlers_.end(); ++it)
						{
							if(*it == cmd.handler) {
								async_comm_handlers_.erase(it);
								break;
							}
						}
						break;
					}
					pthread_mutex_lock(&d_->thread_sync_);
				}
				pthread_mutex_unlock(&d_->thread_sync_);
			}
			if(alltoall_finished_ && async_comm_handlers_.size() == 0) {
				// finished
				break;
			}

			if(alltoall_finished_ == false) {
				alltoall_finished_ = comm_->probe();
			}

			for(int i = 0; i < (int)async_comm_handlers_.size(); ++i) {
				async_comm_handlers_[i]->probe();
			}
		} // while(true)

	}
#if PROFILING_MODE
	void submit_prof_info(int number) {
		comm_time_.submit("comm_thread_task_proc", number);
	}
#endif
private:

	enum COMM_COMMAND {
		SEND,
		MANUAL_CMD,
		ADD_HANDLER,
		REMOVE_HANDLER,
	};

	struct InternalCommand {
		COMM_COMMAND kind;
		union {
			struct {
				// SEND
				CommunicationBuffer* data;
				int target;
			} send;
			// COMM_COMMAND
			CommCommand* cmd;
			AsyncCommHandler* handler;
		};
	};

	struct DynamicDataSet {
		// lock topology
		// FoldNode::send_mutex -> thread_sync_
		pthread_mutex_t thread_sync_;

		// monitor : thread_sync_
		volatile bool command_active_;
		std::deque<InternalCommand> command_queue_;
	} *d_;

	AlltoallCommunicator* comm_;
	std::vector<AsyncCommHandler*> async_comm_handlers_;
	AlltoallBufferHandler* buffer_provider_;
	int buffer_size_;
	bool alltoall_finished_;
	int comm_size_;
	int send_queue_limit_;

	int node_list_length_;
	CommTarget* node_;
	FiberManager* fiber_man_;

	PROF(profiling::TimeSpan comm_time_);

	template <bool proc>
	CommunicationBuffer* get_send_buffer() {
#if 0
		PROF(profiling::TimeKeeper tk_wait);
		PROF(profiling::TimeSpan ts_proc);
		while(d_->total_send_queue_ > s_.comm_size * s_.send_queue_limit) if(proc) {
			PROF(profiling::TimeKeeper tk_proc);
			while(fiber_man_->process_task(1)) ; // process recv task
			PROF(ts_proc += tk_proc);
		}
		PROF(comm_time_ += profiling::TimeSpan(tk_wait) - ts_proc);
#endif // #if 0
		return buffer_provider_->alloc_buffer();
	}

	bool pop_command(InternalCommand* cmd) {
		if(d_->command_queue_.size()) {
			*cmd = d_->command_queue_[0];
			d_->command_queue_.pop_front();
			return true;
		}
		d_->command_active_ = false;
		return false;
	}

	void put_command(InternalCommand& cmd)
	{
		pthread_mutex_lock(&d_->thread_sync_);
		d_->command_queue_.push_back(cmd);
		d_->command_active_ = true;
		pthread_mutex_unlock(&d_->thread_sync_);
	}

	/**
	 * This function does not reset fille_size_ and reserved_size_
	 * because they are very sensitive and it is easy to generate race condition.
	 */
	void send_submit(int target)
	{
		CommTarget& node = node_[target];
		node.cur_buf->length_ = node.filled_size_;

		InternalCommand cmd;
		cmd.kind = SEND;
		cmd.send.data = node.cur_buf;
		cmd.send.target = target;
		put_command(cmd);

		node.cur_buf = NULL;
	}
};

#endif /* ABSTRACT_COMM_HPP_ */

