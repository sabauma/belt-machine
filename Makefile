
all: jit no-jit

jit:
	../pycket/pypy/rpython/bin/rpython -Ojit belt.py

no-jit:
	../pycket/pypy/rpython/bin/rpython -O2   belt.py
