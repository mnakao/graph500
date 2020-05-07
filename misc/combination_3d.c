#include <stdio.h>
#define MAX_SIZE 48
#define MIN_SIZE 12
#define PROCS 512
#define R 32
#define C 16
int main()
{
  for(int X=2;X<MAX_SIZE;X++)
	for(int Y=2;Y<MAX_SIZE;Y++)
	  for(int Z=2;Z<MAX_SIZE;Z++)
		if(X*Y*Z == PROCS && (!(X > MIN_SIZE && Y > MIN_SIZE && Z > MIN_SIZE)))
		  for(int D=2;D<Y;D++)
			if(Y%D == 0 && X * Y /D == C && D * Z == R)
			  printf("%d %d %d %d\n", X, Y, Z, D);
 
  return 0;
}
