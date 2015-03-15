
all: jit no-jit

jit:
	../pypy/rpython/bin/rpython -Ojit belt.py

no-jit:
	../pypy/rpython/bin/rpython -O2   belt.py
