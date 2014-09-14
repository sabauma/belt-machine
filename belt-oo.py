

from rpython.rlib import streamio
from rpython.rlib import jit, debug, objectmodel
from rpython.rlib import rstring

BELT_LEN = 9

class Done(Exception):
    def __init__(self, vals):
        self.values = vals

class ActivationRecord(object):
    _immutable_fields_ = ["pc", "belt", "prev"]
    def __init__(self, pc, belt, prev):
        self.pc   = pc
        self.belt = belt
        self.prev = prev

class Instruction(object):
    def __init__(self):
        raise Exception("Abstract base class")

    def interpret(self, pc, belt, stack):
        raise Exception("Abstract base class")

    def tostring(self):
        return str(self)

class Copy(Instruction):
    _immutable_fields_ = ["value"]
    def __init__(self, val):
        self.value = val

    @jit.unroll_safe
    def interpret(self, pc, belt, stack):
        belt.put(belt.get(self.value))
        return pc + 1, belt, stack, False

    def tostring(self):
        return "COPY %d" % self.value

class Const(Instruction):
    _immutable_fields_ = ["value"]
    def __init__(self, val):
        self.value = val

    @jit.unroll_safe
    def interpret(self, pc, belt, stack):
        belt.put(self.value)
        return pc + 1, belt, stack, False

    def tostring(self):
        return "CONST %d" % self.value

class Call(Instruction):
    _immutable_fields_ = ["destination", "args"]
    def __init__(self, destination, args):
        self.destination = destination
        self.args        = [i for i in reversed(args)]

    @jit.unroll_safe
    def interpret(self, pc, belt, stack):
        stack    = ActivationRecord(pc + 1, belt, stack)
        new_belt = Belt()
        target   = belt.get(self.destination)
        for i in self.args:
            new_belt.put(belt.get(i))
        return target, new_belt, stack, target < pc

    def tostring(self):
        args = " ".join(["B%d" % i for i in reversed(self.args)])
        return "CALL B%d %s" % (self.destination, args)

class Jump(Instruction):
    _immutable_fields_ = ["destination"]
    def __init__(self, destination):
        self.destination = destination

    @jit.unroll_safe
    def interpret(self, pc, belt, stack):
        target = belt.get(self.destination)
        return target, belt, stack, target < pc

    def tostring(self):
        return "JUMP B%d" % self.destination

class Return(Instruction):
    _immutable_fields_ = ["results"]
    def __init__(self, results):
        self.results = [i for i in reversed(results)]

    @jit.unroll_safe
    def interpret(self, pc, belt, stack):
        if stack is None:
            raise Done([belt.get(i) for i in reversed(self.results)])
        for i in self.results:
            stack.belt.put(belt.get(i))
        return stack.pc, stack.belt, stack.prev, False

    def tostring(self):
        args = " ".join(["B%d" % i for i in reversed(self.results)])
        return "RETURN %s" % args

class Binop(Instruction):
    _immutable_fields_ = ["lhs", "rhs"]
    name = "BINOP"
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def binop(v0, v1):
        raise Exception("abstract method")

    @jit.unroll_safe
    def interpret(self, pc, belt, stack):
        v0 = belt.get(self.lhs)
        v1 = belt.get(self.rhs)
        belt.put(self.binop(v0, v1))
        return pc + 1, belt, stack, False

    def tostring(self):
        return "%s B%d B%d" % (self.name, self.lhs, self.rhs)

class Add(Binop):
    name = "ADD"

    @staticmethod
    def binop(v0, v1):
        return v0 + v1

class Sub(Binop):
    name = "SUB"

    @staticmethod
    def binop(v0, v1):
        return v0 - v1

class Lte(Binop):
    name = "LTE"

    @staticmethod
    def binop(v0, v1):
        return 1 if v0 <= v1 else 0

class Pick(Instruction):
    _immutable_fields_ = ["pred", "cons", "alt"]
    def __init__(self, pred, cons, alt):
        self.pred = pred
        self.cons = cons
        self.alt  = alt

    @jit.unroll_safe
    def interpret(self, pc, belt, stack):
        belt.put(belt.get(self.cons) if belt.get(self.pred) != 0 else belt.get(self.alt))
        return pc + 1, belt, stack, False

    def tostring(self):
        return "PICK B%d B%d B%d" % (self.pred, self.cons, self.alt)

def get_labels(instructions):
    labels = {}
    i = 0
    for ins in instructions:
        if not ins:
            continue
        if ins[-1] == ':':
            labels[ins[:-1]] = i
        else:
            i += 1
    return labels

def convert_arg(val, labels):
    val = rstring.strip_spaces(val)
    if val in labels:
        return labels[val]
    val = val[1:] if val[0] == 'b' or val[0] == 'B' else val
    int_rep = int(val, 10)
    return int_rep

def parse(input):
    program = []
    lines   = [rstring.strip_spaces(i) for i in rstring.split(input, '\n')]
    labels  = get_labels(lines)
    for line in lines:
        line = [i for i in rstring.split(line, ' ') if i]
        if not line:
            continue
        ins, args = line[0].lower(), [convert_arg(i, labels) for i in line[1:]]
        if ins[-1] == ':':
            continue
        elif ins == "copy":
            val = Copy(args[0])
        elif ins == "const":
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

def get_printable_location(pc, belt_start, program):
    if pc is None or program is None:
        return "Greens are None"
    return "%d: %s" % (pc, program[pc].tostring())

driver = jit.JitDriver(reds=["stack", "belt"],
                       greens=["pc", "belt_start", "program"],
                       get_printable_location=get_printable_location)

class Belt(object):
    def __init__(self):
        self.start = 0
        self.data  = [0] * BELT_LEN

    @jit.unroll_safe
    def get(self, idx):
        jit.promote(self.start)
        index = (self.start - idx) % BELT_LEN
        return self.data[index]

    # An operation to ensure that all the data is shifted before a trace (hopefully).
    # The hope is to ensure all traces have the same value for the `start` field,
    # as it is used to conver the temporal indexing of the belt to spatial indexing
    # that the jit can more easily work with.
    @jit.unroll_safe
    def reset(self):
        if self.start == 0:
            return
        for i in range(BELT_LEN):
            j = (self.start + i) % BELT_LEN
            self.data[i], self.data[j] = self.data[j], self.data[i]
        self.start = 0

    @jit.unroll_safe
    def put(self, val):
        jit.promote(self.start)
        self.start = (self.start + 1) % BELT_LEN
        self.data[self.start] = val

def main_loop(program):
    pc    = 0
    belt  = Belt()
    stack = None
    try:
        while True:
            driver.jit_merge_point(pc=pc, belt_start=belt.start, program=program, belt=belt, stack=stack)
            ins = program[pc]
            pc, belt, stack, can_enter = ins.interpret(pc, belt, stack)
            if can_enter:
                driver.can_enter_jit(pc=pc, belt_start=belt.start, program=program, belt=belt, stack=stack)
    except Done as d:
        print "Results: ", d.values

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
        driver.exe_name = 'belt-oo-%(backend)s'
    else:
        driver.exe_name = 'belt-oo-%(backend)s-nojit'
    return entry_point, None

if __name__ == "__main__":
    import sys
    entry_point(sys.argv)
