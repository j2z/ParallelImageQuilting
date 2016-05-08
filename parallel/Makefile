#SRCLIST = image_data main
SRCLIST = image_data color seam_carve update_map imagequilt_serial main
#SRCLIST = polar_transform patch_copy sample error_map main
EXECNAME = ImageQuilt
CUDASRCLIST = imagequilt_cuda color_cu error_cu polar_transform_cu


SRCDIR=src
INCDIR=../includes
OBJDIR=obj

CC=g++ -m64

CFLAGS=-I $(INCDIR) -O3 -Wall
LFLAGS=-L/usr/X11R6/lib
LIBS=-lm -lpthread -lX11 -L/usr/local/depot/cuda-6.5/lib64/ -lcudart
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
