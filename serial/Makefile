SRCLIST = imagequilt_serial
EXECNAME = out


SRCDIR=src
INCDIR=../includes
OBJDIR=obj

CC=g++ -m64

CFLAGS=-I $(INCDIR) -O3 -Wall
LFLAGS=-L/usr/X11R6/lib
LIBS=-lm -lpthread -lX11

OBJS = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(SRCLIST)))


all: $(EXECNAME)

$(EXECNAME) : $(OBJS)
	$(CC) $(LFLAGS) $(LIBS) -o $(EXECNAME) $(OBJS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CC) -c $(CFLAGS) $< -o $@

$(OBJDIR):
	mkdir $(OBJDIR)

clean:
	rm -rf $(OBJDIR)
	rm -f $(EXECNAME)
