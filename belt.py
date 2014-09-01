
from rpython.rlib import streamio
from rpython.rlib import jit, debug, objectmodel
from rpython.rlib import rstring

BELT_LEN = 8

class ActivationRecord(object):
    _immutable_fields_ = ["prev", "pc", "belt"]
    def __init__(self, prev, pc, belt):
        self.prev = prev
        self.pc   = pc
        self.belt = belt

class Instruction(object):
    pass

class Const(Instruction):
    _immutable_fields_ = ["value"]
    def __init__(self, val):
        self.value = val

class Call(Instruction):
    _immutable_fields_ = ["destination", "args[*]"]
    def __init__(self, destination, args):
        self.destination = destination
        self.args        = args

class Jump(Instruction):
    _immutable_fields_ = ["destination"]
    def __init__(self, destination):
        self.destination = destination

class Return(Instruction):
    _immutable_fields_ = ["results[*]"]
    def __init__(self, results):
        self.results = results

class Binop(Instruction):
    _immutable_fields_ = ["lhs", "rhs"]
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

class Add(Binop):
    pass

class Sub(Binop):
    pass

class Lte(Binop):
    pass

class Pick(Instruction):
    _immutable_fields_ = ["pred", "cons", "alt"]
    def __init__(self, pred, cons, alt):
        self.pred = pred
        self.cons = cons
        self.alt  = alt

#program = [Const(10), Const(0), Pick(0, 0, 1), Return([0])]

def parse(input):
    program = []
    lines = rstring.split(input, '\n')
    for line in lines:
        line = [i for i in rstring.split(line, ' ') if i]
        if not line:
            continue
        ins, args = line[0], [int(i, 10) for i in line[1:]]
        if ins == "const":
            val = Const(args[0])
        elif ins == "call":
            val = Call(args[0], args[1:])
        elif ins == "jump":
            val = Jump(args[0])
        elif ins == "return":
            val = Return(args)
        elif ins == "add":
            val = Add(args[0], args[1])
        elif ins ==  "sub":
            val = Sub(args[0], args[1])
        elif ins == "lte":
            val = Lte(args[0], args[1])
        elif ins == "pick":
            val = Pick(args[0], args[1], args[2])
        else:
            raise Exception("Unparsable instruction %s" % ins)
        program.append(val)
    return program[:]

#t0 = """
#ENTRY:
    #CONST FIB
    #CONST 35
    #CALL B1  B0
    #RETURN B0
#FIB:
    #CONST 1
    #CONST DONE
    #CONST NEXT
    #LTE B3  B2      ;; n <= 1
    #PICK B0  B2  B1 ;; pick our jump destination
    #JUMP B0         ;; if (n <= 1) { goto DONE } { goto NEXT }
#DONE:
    #RETURN B4
#NEXT:
    #SUB B5  B4   ;; n-1
    #SUB B0  B5   ;; n-2
    #CONST FIB
    #CALL B0  B2 ;; fib(n-1)
    #CALL B1  B2 ;; fib(n-2)
    #ADD B0  B1
    #RETURN B0
#"""

#test = [
    #Const(30),
    #Const(4),
    #Call(0, [1]),
    #Return([0]),
    #Const(1),
    #Const(10),
    #Const(11),
    #Lte(3, 2),
    #Pick(0, 2, 1),
    #Jump(0),
    #Return([4]),
    #Sub(5, 4),
    #Sub(0, 5),
    #Const(4),
    #Call(0, [2]),
    #Call(1, [2]),
    #Add(0, 1),
    #Return([0])
    #]

driver = jit.JitDriver(reds=["stack", "belt"], greens=["pc", "program"])

class Belt(object):
    _immutable_fields_ = ["length", "data"]
    def __init__(self, len):
        self.start  = 0
        self.data   = [0] * len
        self.length = len
    def get(self, idx):
        assert idx < self.length
        return self.data[self.start - idx]
    def put(self, val):
        self.start = (self.start + 1) % self.length
        self.data[self.start] = val

def main_loop(program):
    pc    = 0
    belt  = Belt(BELT_LEN)
    stack = None
    while True:
        driver.jit_merge_point(pc=pc, program=program, belt=belt, stack=stack)
        ins = program[pc]
        typ = type(ins)
        if typ is Const:
            belt.put(ins.value)
            pc += 1
        elif typ is Call:
            stack    = ActivationRecord(stack, pc + 1, belt)
            new_belt = Belt(BELT_LEN)
            target   = belt.get(ins.destination)
            for i in reversed(ins.args):
                new_belt.put(belt.get(i))
            can_enter = target < pc
            belt = new_belt
            pc = target
            if can_enter:
                driver.can_enter_jit(pc=pc, program=program, belt=belt, stack=stack)
        elif typ is Jump:
            target = belt.get(ins.destination)
            can_enter = target < pc
            pc = target
            if can_enter:
                driver.can_enter_jit(pc=pc, program=program, belt=belt, stack=stack)
        elif typ is Return:
            if stack is None:
                print "Results", [belt.get(i) for i in ins.results]
                break
            for i in reversed(ins.results):
                stack.belt.put(belt.get(i))
            pc, belt, stack = stack.pc, stack.belt, stack.prev
        elif typ is Add:
            v0 = belt.get(ins.lhs)
            v1 = belt.get(ins.rhs)
            belt.put(v0 + v1)
            pc += 1
        elif typ is Sub:
            v0 = belt.get(ins.lhs)
            v1 = belt.get(ins.rhs)
            belt.put(v0 - v1)
            pc += 1
        elif typ is Lte:
            v0 = belt.get(ins.lhs)
            v1 = belt.get(ins.rhs)
            belt.put(1 if v0 <= v1 else 0)
            pc += 1
        elif typ is Pick:
            belt.put(belt.get(ins.cons) if belt.get(ins.pred) != 0 else belt.get(ins.alt))
            pc += 1
        else:
            raise Exception("Unimplemented opcode")

def readfile_rpython(fname):
    f = streamio.open_file_as_stream(fname)
    s = f.readall()
    f.close()
    return s

def run(fname):
    contents = readfile_rpython(fname)
    program = parse(contents)
    main_loop(program)

def entry_point(argv):
    try:
        filename = argv[1]
    except IndexError:
        print "You must supply a filename"
        return 1
    run(filename)
    return 0

def target(driver, args):
    if driver.config.translation.jit:
        driver.exe_name = 'belt-%(backend)s'
    else:
        driver.exe_name = 'belt-%(backend)s-nojit'
    return entry_point, None

if __name__ == "__main__":
    import sys
    entry_point(sys.argv)
