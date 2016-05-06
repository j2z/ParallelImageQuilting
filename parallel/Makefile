SRCLIST = polar_transform patch_copy sample error_map copytest
EXECNAME = out
CUDASRCLIST =


SRCDIR=src
INCDIR=../includes
OBJDIR=obj

CC=g++ -m64

CFLAGS=-I $(INCDIR) -O3 -Wall
LFLAGS=-L/usr/X11R6/lib
#LIBS=-lm -lpthread -lX11 -L/usr/local/depot/cuda-6.5/lib64/ -lcudart
LIBS=-lm -lpthread -lX11 -L/usr/local/depot/cuda-6.5/lib64/
NVCC=nvcc -arch=sm_20
NVCCFLAGS=-O3 -m64

OBJS = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(SRCLIST)))
CUDAOBJS = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(CUDASRCLIST)))


all: $(EXECNAME)

$(EXECNAME) : $(OBJS) $(CUDAOBJS)
	$(CC) $(LFLAGS) $(LIBS) -o $(EXECNAME) $(OBJS) $(CUDAOBJS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CC) -c $(CFLAGS) $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

$(OBJDIR):
	mkdir $(OBJDIR)

clean:
	rm -rf $(OBJDIR)
	rm -f $(EXECNAME)
