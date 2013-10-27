/*
 * fiber.hpp
 *
 *  Created on: Mar 6, 2012
 *      Author: koji
 */

#ifndef FIBER_HPP_
#define FIBER_HPP_

#include <pthread.h>

#include <deque>
#include <vector>

class Runnable
{
public:
	virtual ~Runnable() { }
	virtual void run() = 0;
};

class FiberManager
{
public:
	static const int MAX_PRIORITY = 4;

	FiberManager()
		: command_active_(false)
		, terminated_(false)
		, suspended_(0)
		, max_priority_(0)
	{
		pthread_mutex_init(&thread_sync_, NULL);
		pthread_cond_init(&thread_state_,  NULL);
		cleanup_ = false;
	}

	~FiberManager()
	{
		if(!cleanup_) {
			cleanup_ = true;
			pthread_mutex_destroy(&thread_sync_);
			pthread_cond_destroy(&thread_state_);
		}
	}

	void begin_processing()
	{
		terminated_ = false;
	}

	void enter_processing()
	{
		// command loop
		while(true) {
			if(command_active_) {
				pthread_mutex_lock(&thread_sync_);
				Runnable* cmd;
				while(pop_command(&cmd, 0)) {
					pthread_mutex_unlock(&thread_sync_);
					cmd->run();
					pthread_mutex_lock(&thread_sync_);
				}
				pthread_mutex_unlock(&thread_sync_);
			}
			if(command_active_ == false) {
				pthread_mutex_lock(&thread_sync_);
				if(command_active_ == false) {
					if( terminated_ ) { pthread_mutex_unlock(&thread_sync_); break; }
					++suspended_;
#if PROFILING_MODE
					profiling::TimeKeeper wait_;
#endif
					pthread_cond_wait(&thread_state_, &thread_sync_);
#if PROFILING_MODE
					wait_time_ += wait_;
#endif
					--suspended_;
				}
				pthread_mutex_unlock(&thread_sync_);
			}
		}
	}

	bool process_task(int priority_lower_bound) {
		if(command_active_) {
			pthread_mutex_lock(&thread_sync_);
			Runnable* cmd;
			if(pop_command(&cmd, priority_lower_bound)) {
				pthread_mutex_unlock(&thread_sync_);
				cmd->run();
				return true;
			}
			pthread_mutex_unlock(&thread_sync_);
		}
		return false;
	}

	void end_processing()
	{
		pthread_mutex_lock(&thread_sync_);
		terminated_ = true;
		pthread_mutex_unlock(&thread_sync_);
		pthread_cond_broadcast(&thread_state_);
	}

	void submit(Runnable* r, int priority)
	{
		pthread_mutex_lock(&thread_sync_);
		command_active_ = true;
		command_queue_[priority].push_back(r);
		max_priority_ = std::max(priority, max_priority_);
		int num_suspended = suspended_;
		pthread_mutex_unlock(&thread_sync_);
		if(num_suspended > 0) pthread_cond_broadcast(&thread_state_);
	}

	template <typename T>
	void submit_array(T* runnable_array, size_t length, int priority)
	{
		pthread_mutex_lock(&thread_sync_);
		command_active_ = true;
		std::deque<Runnable*>& queue = command_queue_[priority];
		size_t pos = queue.size();
		queue.insert(queue.end(), length, NULL);
		for(size_t i = 0; i < length; ++i) {
			queue[pos + i] = &runnable_array[i];
		}
		max_priority_ = std::max(priority, max_priority_);
		int num_suspended = suspended_;
		pthread_mutex_unlock(&thread_sync_);
		if(num_suspended > 0) pthread_cond_broadcast(&thread_state_);
	}
#if PROFILING_MODE
	void submit_wait_time(const char* content, int number) {
		wait_time_.submit(content, number);
	}
	void reset_wait_time() {
		wait_time_.reset();
	}
#endif
private:
	//
	bool cleanup_;
	pthread_mutex_t thread_sync_;
	pthread_cond_t thread_state_;

	volatile bool command_active_;

	bool terminated_;
	int suspended_;
	int max_priority_;

	std::deque<Runnable*> command_queue_[MAX_PRIORITY];
#if PROFILING_MODE
	profiling::TimeSpan wait_time_;
#endif

	bool pop_command(Runnable** cmd, int priority_lower_bound) {
		int i = max_priority_ + 1;
		while(i-- > priority_lower_bound) {
			assert (i < MAX_PRIORITY);
			if(command_queue_[i].size()) {
				*cmd = command_queue_[i][0];
				command_queue_[i].pop_front();
				max_priority_ = i;
				return true;
			}
		}
		max_priority_ = 0;
		command_active_ = false;
		return false;
	}
};

template <typename T>
class ConcurrentStack
{
public:
	ConcurrentStack(int limit)
		: limit_(limit)
	{
		pthread_mutex_init(&thread_sync_, NULL);
		pthread_cond_init(&thread_state_,  NULL);
		cleanup_ = false;
	}

	~ConcurrentStack()
	{
		if(!cleanup_) {
			cleanup_ = true;
			pthread_mutex_destroy(&thread_sync_);
			pthread_cond_destroy(&thread_state_);
		}
	}

	void push(const T& d)
	{
		pthread_mutex_lock(&thread_sync_);
		while((int)stack_.size() >= limit_) {
			pthread_cond_wait(&thread_state_, &thread_sync_);
		}
		int old_size = (int)stack_.size();
		stack_.push_back(d);
		pthread_mutex_unlock(&thread_sync_);
		if(old_size == 0) pthread_cond_broadcast(&thread_state_);
	}

	T pop()
	{
		pthread_mutex_lock(&thread_sync_);
		while(stack_.size() == 0) {
			pthread_cond_wait(&thread_state_, &thread_sync_);
		}
		int old_size = (int)stack_.size();
		T r = stack_.back(); stack_.pop_back();
		pthread_mutex_unlock(&thread_sync_);
		if(old_size == limit_) pthread_cond_broadcast(&thread_state_);
		return r;
	}

private:
	int limit_;
	bool cleanup_;
	std::vector<T> stack_;
	pthread_mutex_t thread_sync_;
	pthread_cond_t thread_state_;
};


#endif /* FIBER_HPP_ */
