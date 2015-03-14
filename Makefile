
all: jit no-jit

jit:
	../pypy/rpython/bin/rpython -Ojit belt-oo.py

no-jit:
	../pypy/rpython/bin/rpython -O2   belt-oo.py
