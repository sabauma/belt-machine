
import operator as op
from   rpython.rlib          import streamio
from   rpython.rlib          import jit, debug, objectmodel, unroll
from   rpython.rlib          import rstring
from   rpython.rlib.listsort import TimSort

BELT_LEN = 8

class Done(Exception):
    def __init__(self, vals):
        self.values = vals

class Instruction(object):

    _immutable_fields_ = ['loop_header']

    loop_header = False

    def __init__(self):
        raise Exception("Abstract base class")

    def interpret(self, pc, belt, stack):
        raise Exception("Abstract base class")

    def set_loop_header(self):
        self.loop_header = True

    def tostring(self):
        return str(self)

class Copy(Instruction):
    _immutable_fields_ = ["value"]
    def __init__(self, val):
        assert 0 <= val < BELT_LEN
        self.value = val

    def interpret(self, pc, belt, stack):
        belt.put(belt.get(self.value))
        return pc + 1, stack, -1

    def tostring(self):
        return "COPY %d" % self.value

class Const(Instruction):
    _immutable_fields_ = ["value"]
    def __init__(self, val):
        self.value = val

    def interpret(self, pc, belt, stack):
        belt.put(self.value)
        return pc + 1, stack, -1

    def tostring(self):
        return "CONST %d" % self.value

class Call(Instruction):
    _immutable_fields_ = ["destination", "args[*]", "saves"]
    def __init__(self, destination, saves, args):
        self.destination = destination
        self.args        = [i for i in reversed(args)]
        self.saves       = saves

    @jit.unroll_safe
    def interpret(self, pc, belt, stack):
        target = jit.promote(belt.get(self.destination))
        # stack  = ActivationRecord(pc + 1, belt, stack)
        stack  = make_activation_record(self.saves, pc + 1, belt, stack)
        vals   = [belt.get(i) for i in self.args]
        belt.reset()
        for v in vals:
            belt.put(v)
        return target, stack, -1

    def tostring(self):
        args = " ".join(["B%d" % i for i in reversed(self.args)])
        return "CALL B%d %s" % (self.destination, args)

class Jump(Instruction):
    _immutable_fields_ = ["destination"]
    def __init__(self, destination):
        self.destination = destination

    def interpret(self, pc, belt, stack):
        target = jit.promote(belt.get(self.destination))
        return target, stack, -1

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
        stack.restore(belt)
        for v in vals:
            belt.put(v)
        ret = jit.promote(stack.pc)
        return ret, stack.prev, -1

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

    def interpret(self, pc, belt, stack):
        v0 = belt.get(self.lhs)
        v1 = belt.get(self.rhs)
        belt.put(self.binop(v0, v1))
        return pc + 1, stack, -1

    def tostring(self):
        return "%s B%d B%d" % (self.name, self.lhs, self.rhs)

def make_binop(ins_name, op):
    class BinopImp(Binop):
        name = ins_name
        @staticmethod
        def binop(v0, v1):
            return op(v0, v1)
    BinopImp.__name__ += "ins_name"
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
        return pc + 1, stack, -1

    def tostring(self):
        return "PICK B%d B%d B%d" % (self.pred, self.cons, self.alt)

class IndirectionInstruction(Instruction):

    _immutable_fields_ = ['inner']

    def __init__(self, inner):
        self.inner = inner

    def interpret(self, pc, belt, stack):
        return self.inner.interpret(pc, belt, stack)

    def direct_interpret(self, pc, belt, stack):
        return self.inner.interpret(pc, belt, stack)

class DispatchNode(IndirectionInstruction):

    _immutable_fields_ = ['alternatives[*]']

    def __init__(self, inner):
        IndirectionInstruction.__init__(self, inner)
        self.loop_header = True
        self.alternatives = []

    @jit.elidable_promote('all')
    def get_alternative(self, alt):
        return self.alternatives[alt]

    def add_alternative(self, alt):
        idx = len(self.alternatives)
        self.alternatives = self.alternatives + [alt]
        return idx

    def perform_dispatch(self, index, pc, stack):
        alt = self.get_alternative(index)
        return alt, stack, index

class CaseNode(IndirectionInstruction):

    _immutable_fields_ = ['index', 'dispatch_node']

    def __init__(self, inner, index, dispatch_node):
        IndirectionInstruction.__init__(self, inner)
        inner.loop_header = False
        self.index = index
        self.dispatch_node = dispatch_node

    def interpret(self, pc, belt, stack):
        return self.dispatch_node, stack, self.index

    def set_loop_header(self):
        pass

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
            val = Call(args[0], args[1], args[2:])
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

driver = jit.JitDriver(reds=["dispatch_index", "stack", "belt", "flowgraph"],
                       greens=["pc", "program"],
                       virtualizables=["belt"],
                       get_printable_location=get_printable_location,
                       should_unroll_one_iteration=lambda *args: True)

def make_belt(LEN):
    attrs    = ["_elem_%d_of_%d" % (d, LEN) for d in range(LEN)]
    unrolled = unroll.unrolling_iterable(enumerate(attrs))
    swaps    = unroll.unrolling_iterable(zip(attrs[-2::-1], attrs[-1::-1]))

    class Belt(object):
        _virtualizable_ = attrs
        def __init__(self):
            self = jit.hint(self, access_directly=True, fresh_virtualizable=True)
            self.reset()

        def get(self, idx):
            for i, attr in unrolled:
                if i == idx:
                    return getattr(self, attr)
            assert False, "belt index error"

        def put(self, val):
            for src, target in swaps:
                setattr(self, target, getattr(self, src))
            setattr(self, attrs[0], val)

        def reset(self):
            for _, attr in unrolled:
                setattr(self, attr, 0)

    class AbstractActivationRecord(object):
        _immutable_fields_ = ["pc", "prev"]

        def __init__(self, pc, belt, prev):
            self.pc = pc
            self.prev = prev

    def make_activation_record(size):

        ar_attrs = unroll.unrolling_iterable(list(attrs)[:size])

        class ActivationRecord(AbstractActivationRecord):
            _immutable_fields_ = attrs
            def __init__(self, pc, belt, prev):
                AbstractActivationRecord.__init__(self, pc, belt, prev)
                for attr in ar_attrs:
                    setattr(self, attr, getattr(belt, attr))

            def restore(self, belt):
                for attr in ar_attrs:
                    setattr(belt, attr, getattr(self, attr))

        ActivationRecord.__name__ += "Size%d" % size
        return ActivationRecord

    record_classes = [make_activation_record(i) for i in range(LEN)]

    def make_ar(size, pc, belt, prev):
        cls = record_classes[size]
        return cls(pc, belt, prev)

    return Belt, make_ar

Belt, make_activation_record = make_belt(BELT_LEN)

binops = unroll.unrolling_iterable(Binop.classes.values())

EMPTY = {}

class Flowgraph(object):

    _immutable_fields_ = ['_successors', '_predecessors', 'program']

    def __init__(self, program):
        self._successors   = {}
        self._predecessors = {}
        self.program       = program

    def add_edge(self, source, destination):
        succ = self.successors(source)
        # pred = self.predecessors(destination)
        if destination not in succ:
            succ[destination] = True
            self.find_loops()
            self.component_analysis()

        # if source not in pred:
            # pred[source] = True

    def component_analysis(self):
        scc = SCC(self._successors)
        scc.compute_scc()
        components = scc.scc_set

        for component in components:
            if len(component) == 1:
                continue
            headers = [i for i in component if self.program[i].loop_header]
            TimSort(headers).sort()
            if len(headers) <= 1:
                continue

            master, subs = headers[0], headers[1:]
            master_ins = self.program[master]
            if not isinstance(master_ins, DispatchNode):
                self.program[master] = master_ins = DispatchNode(master_ins)
            for sub in subs:
                sub_ins = self.program[sub]
                if not isinstance(sub_ins, CaseNode):
                    idx = master_ins.add_alternative(sub)
                    self.program[sub] = sub_ins = CaseNode(sub_ins, idx, master)
                else:
                    assert sub_ins.dispatch_node == master

    def successors(self, node):
        links = self._successors.get(node, None)
        if links is None:
            self._successors[node] = links = {}
        return links

    def predecessors(self, node):
        links = self._predecessors.get(node, None)
        if links is None:
            self._predecessors[node] = links = {}
        return links

    # Perform a depth first traversal to find back edges
    def find_loops(self):
        if jit.we_are_jitted():
            return
        seen = {}
        todo = [0]
        while todo:
            node = todo.pop()
            if node in seen:
                self.program[node].set_loop_header()
                # self.program[node].loop_header = True
                # print "marking loop_header %d: %s" % (node, self.program[node].tostring())
                continue
            seen[node] = True
            for succ in self.successors(node):
                todo.append(succ)

    def print_form(self):
        output = []
        for i, ins in enumerate(self.program):
            output.append("%d: %s goes to" % (i, ins.tostring()))
            for succ in self.successors(i):
                succ_ins = self.program[succ]
                output.append("    %d: %s" % (succ, succ_ins.tostring()))
        return "\n".join(output)

class VertexData(object):

    __slots__ = ('vertex', 'index', 'lowlink', 'onstack')

    def __init__(self, vertex, index, lowlink):
        self.vertex  = vertex
        self.index   = index
        self.lowlink = lowlink
        self.onstack = False

class SCC(object):

    def __init__(self, graph):
        self.graph   = graph
        self.index   = 0
        self.S       = []
        self.data    = {}
        self.scc_set = []

    def lookup_vertex(self, v):
        data = self.data.get(v, None)
        if data is None:
            data = VertexData(v, -1, -1)
            self.data[v] = data
        return data

    def strongconnect(self, v):
        vertex_v         = self.lookup_vertex(v)
        vertex_v.index   = self.index
        vertex_v.lowlink = self.index
        self.index += 1
        self.S.append(vertex_v)
        vertex_v.onstack = True

        # For each successor of v
        for w in self.graph.get(v, EMPTY):
            vertex_w = self.lookup_vertex(w)
            if vertex_w.index == -1:
                self.strongconnect(w)
                vertex_v.lowlink = min(vertex_v.lowlink, vertex_w.lowlink)
            elif vertex_w.onstack:
                vertex_v.lowlink = min(vertex_v.lowlink, vertex_w.index)

        if vertex_v.lowlink == vertex_v.index:
            newscc = {}
            while True:
                w = self.S.pop()
                w.onstack = False
                newscc[w.vertex] = None
                if w is vertex_v:
                    break
            self.scc_set.append(newscc)

    def compute_scc(self):
        for v in self.graph.iterkeys():
            if v is None:
                continue
            vertex_v = self.lookup_vertex(v)
            if vertex_v.index == -1:
                self.strongconnect(v)

@jit.elidable_promote('all')
def get_instruction(program, index):
    return program[index]

def main_loop(program):
    pc    = 0
    belt  = Belt()
    stack = None
    dispatch_index = -1
    flowgraph = Flowgraph(program)
    try:
        while True:
            driver.jit_merge_point(pc=pc, program=program, dispatch_index=dispatch_index, belt=belt, stack=stack, flowgraph=flowgraph)
            ins = get_instruction(program, pc)
            dispatch_index = jit.promote(dispatch_index)
            if dispatch_index != -1:
                if isinstance(ins, DispatchNode):
                    new_pc, stack, dispatch_index = ins.perform_dispatch(dispatch_index, pc, stack)
                else:
                    assert isinstance(ins, CaseNode)
                    new_pc, stack, dispatch_index = IndirectionInstruction.interpret(ins, pc, belt, stack)
            else:
                new_pc, stack, dispatch_index = ins.interpret(pc, belt, stack)

            if not jit.we_are_jitted():
                flowgraph.add_edge(pc, new_pc)
            pc = new_pc
            if get_instruction(program, pc).loop_header:
                driver.can_enter_jit(pc=pc, program=program, dispatch_index=dispatch_index, belt=belt, stack=stack, flowgraph=flowgraph)
    except Done as d:
        print "Results: ", d.values
        print flowgraph.print_form()

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
    if len(argv) == 3:
        jitargs = argv[1]
        del argv[1]
        jit.set_user_param(None, jitargs)
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

