#ifndef FIBER_HPP_
#define FIBER_HPP_

class Runnable
{
public:
	virtual ~Runnable() { }
	virtual void run() = 0;
};

#endif /* FIBER_HPP_ */
