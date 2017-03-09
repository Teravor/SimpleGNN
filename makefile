CC	= g++
CCFLAGS	= -O3 # -Wall

CCLIBFLAGS = -fpic #-std=c++11 -fexceptions # -msse4.2

LIBPATH = -L/usr/lib/ -L/m/home/home8/82/morooke1/unix/lib

INCPATH	= -I../



#SRC	= ga.c element.c
SRC	= *.cpp 
INC	= $(SRC:.c=.h)
OBJ = $(SRC:.c=.o)

default: all

all: runme.exe

runme.exe: myGa7.cpp 
	$(CC) $(CCFLAGS) $(CCLIBFLAGS)  $(INCPATH) -c myGa7.cpp  -I /m/home/home8/82/morooke1/unix/.local/usr/include/  
	$(CC) $(CCFLAGS) -o runme.exe myGa7.o   $(LIBPATH) -l networks #  -lm 

clean:
	rm -rf *.o *.exe
