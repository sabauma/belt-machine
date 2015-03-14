
import operator as op
from   rpython.rlib import streamio
from   rpython.rlib import jit, debug, objectmodel, unroll
from   rpython.rlib import rstring

BELT_LEN = 16

class Done(Exception):
    def __init__(self, vals):
        self.values = vals

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
        assert 0 <= val < BELT_LEN
        self.value = val

    def interpret(self, pc, belt, stack):
        belt.put(belt.get(self.value))
        return pc + 1, stack, False

    def tostring(self):
        return "COPY %d" % self.value

class Const(Instruction):
    _immutable_fields_ = ["value"]
    def __init__(self, val):
        self.value = val

    def interpret(self, pc, belt, stack):
        belt.put(self.value)
        return pc + 1, stack, False

    def tostring(self):
        return "CONST %d" % self.value

class Call(Instruction):
    _immutable_fields_ = ["destination", "args[*]"]
    def __init__(self, destination, args):
        self.destination = destination
        self.args        = [i for i in reversed(args)]

    @jit.unroll_safe
    def interpret(self, pc, belt, stack):
        stack  = ActivationRecord(pc + 1, belt, stack)
        target = belt.get(self.destination)
        vals   = [belt.get(i) for i in self.args]
        belt.reset()
        for v in vals:
            belt.put(v)
        return target, stack, target < pc

    def tostring(self):
        args = " ".join(["B%d" % i for i in reversed(self.args)])
        return "CALL B%d %s" % (self.destination, args)

class Jump(Instruction):
    _immutable_fields_ = ["destination"]
    def __init__(self, destination):
        self.destination = destination

    def interpret(self, pc, belt, stack):
        target = belt.get(self.destination)
        return target, stack, target < pc

    def tostring(self):
        return "JUMP B%d" % self.destination

class Return(Instruction):
    _immutable_fields_ = ["results[*]"]
    def __init__(self, results):
        self.results = [i for i in reversed(results)]

    @jit.unroll_safe
    def interpret(self, pc, belt, stack):
        if stack is None:
            raise Done([belt.get(i) for i in reversed(self.results)])
        vals = [belt.get(i) for i in self.results]
        belt.restore(stack)
        for v in vals:
            belt.put(v)
        return stack.pc, stack.prev, stack.pc < pc

    def tostring(self):
        args = " ".join(["B%d" % i for i in reversed(self.results)])
        return "RETURN %s" % args

class Binop(Instruction):
    _immutable_fields_ = ["lhs", "rhs"]
    name = "BINOP"

    classes = {}

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    @staticmethod
    def binop(v0, v1):
        raise Exception("abstract method")

    @staticmethod
    def get_cls(name):
        return Binop.classes[name]

    @staticmethod
    def add_cls(cls):
        Binop.classes[cls.name] = cls

    @objectmodel.always_inline
    def interpret(self, pc, belt, stack):
        v0 = belt.get(self.lhs)
        v1 = belt.get(self.rhs)
        belt.put(self.binop(v0, v1))
        return pc + 1, stack, False

    def tostring(self):
        return "%s B%d B%d" % (self.name, self.lhs, self.rhs)

def make_binop(ins_name, op):
    class BinopImp(Binop):
        name = ins_name
        @staticmethod
        def binop(v0, v1):
            return op(v0, v1)
    Binop.add_cls(BinopImp)
    return BinopImp

def make_cmp(ins_name, op):
    class Cmp(Binop):
        name = ins_name
        @staticmethod
        def binop(v0, v1):
            return 1 if op(v0, v1) else 0
    Binop.add_cls(Cmp)
    return Cmp

BINOPS = [("ADD", op.add),
          ("SUB", op.sub),
          ("MUL", op.mul),
          ("DIV", op.div),
          ("MOD", op.mod),
          ]

for i in BINOPS:
    make_binop(*i)

CMPS = [("LTE", op.le),
        ("GTE", op.ge),
        ("EQ" , op.eq),
        ("LT" , op.lt),
        ("GT" , op.gt),
        ("NEQ", op.ne),
        ]

for i in CMPS:
    make_cmp(*i)

class Pick(Instruction):
    _immutable_fields_ = ["pred", "cons", "alt"]
    def __init__(self, pred, cons, alt):
        self.pred = pred
        self.cons = cons
        self.alt  = alt

    def interpret(self, pc, belt, stack):
        belt.put(belt.get(self.cons) if belt.get(self.pred) != 0 else belt.get(self.alt))
        return pc + 1, stack, False

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
        ins, args = line[0].upper(), [convert_arg(i, labels) for i in line[1:]]
        if ins[-1] == ':':
            continue
        elif ins == "COPY":
            val = Copy(args[0])
        elif ins == "CONST":
            val = Const(args[0])
        elif ins == "CALL":
            val = Call(args[0], args[1:])
        elif ins == "JUMP":
            val = Jump(args[0])
        elif ins == "RETURN":
            val = Return(args)
        elif ins == "PICK":
            val = Pick(args[0], args[1], args[2])
        elif ins in Binop.classes:
            val = Binop.get_cls(ins)(args[0], args[1])
        else:
            raise Exception("Unparsable instruction %s" % ins)
        program.append(val)
    return program[:]

def get_printable_location(pc, program):
    if pc is None or program is None:
        return "Greens are None"
    return "%d: %s" % (pc, program[pc].tostring())

driver = jit.JitDriver(reds=["stack", "belt"],
                       greens=["pc", "program"],
                       virtualizables=["belt"],
                       get_printable_location=get_printable_location)

def make_belt(LEN):
    attrs    = ["elem_%d" % d for d in range(LEN)]
    unrolled = unroll.unrolling_iterable(enumerate(attrs))
    swaps    = unroll.unrolling_iterable(zip(attrs[-2::-1], attrs[-1::-1]))

    class Belt(object):
        _virtualizable_ = attrs
        def __init__(self, vals=None):
            self = jit.hint(self, access_directly=True, fresh_virtualizable=True)
            self.reset()

        def get(self, idx):
            jit.promote(idx)
            for i, attr in unrolled:
                if i == idx:
                    return getattr(self, attr)
            raise IndexError

        def put(self, val):
            for src, target in swaps:
                setattr(self, target, getattr(self, src))
            setattr(self, attrs[0], val)

        def reset(self):
            for _, attr in unrolled:
                setattr(self, attr, 0)

        def restore(self, ar):
            for _, attr in unrolled:
                setattr(self, attr, getattr(ar, attr))

    class ActivationRecord(object):
        _immutable_fields_ = ["pc", "prev"] + attrs
        def __init__(self, pc, belt, prev):
            self.pc   = pc
            self.belt = belt
            self.prev = prev
            for i, attr in unrolled:
                setattr(self, attr, getattr(belt, attr))

    return Belt, ActivationRecord

Belt, ActivationRecord = make_belt(BELT_LEN)

def main_loop(program):
    pc    = 0
    belt  = Belt()
    stack = None
    try:
        while True:
            driver.jit_merge_point(pc=pc, program=program, belt=belt, stack=stack)
            ins = program[pc]
            t = type(ins)
            pc, stack, can_enter = ins.interpret(pc, belt, stack)
            if can_enter:
                driver.can_enter_jit(pc=pc, program=program, belt=belt, stack=stack)
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

