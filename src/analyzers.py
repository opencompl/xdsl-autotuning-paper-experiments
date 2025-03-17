import subprocess
from typing import Tuple
from abc import ABC,abstractmethod
from io import StringIO
from dataclasses import dataclass
from xdsl.ir import Operation
from xdsl.dialects.builtin import ModuleOp
from xdsl.backend.assembly_printer import AssemblyPrinter

def run_command(
        command: str,
        timeout: int | None = None,
        input: str | None = None,
) -> Tuple[str,str] :
    output = subprocess.run(
        command,
        shell=True,
        text=True,
        input = input,
        capture_output=True,
        timeout = timeout
    )
    return output.stdout,output.stderr


@dataclass
class Cost:
    execution_time: float

class Analyzer(ABC):

    def consistently_evaluate(self, o:Operation) -> Cost:
        self.check_input_consistency(o)
        c = self.evaluate(o)
        self.check_output_consistency(c)
        return c

    @abstractmethod
    def evaluate(self, o:Operation) -> Cost:
        ...

    @abstractmethod
    def check_input_consistency(self, o:Operation):
        ...

    @abstractmethod
    def check_output_consistency(self, c: Cost):
        ...

class BBAnalyzer(Analyzer):
    
    def check_input_consistency(self, o:Operation):
        # TODO: check if o is a plain-assembly BB
        ...

@dataclass
class LLVM_MCA(BBAnalyzer):

    arch: str = "native"
    microarch: str = "native"

    def check_output_consistency(self, c: Cost):
        ...

    def evaluate(self, o: Operation) -> Cost:

        if not isinstance(o, ModuleOp):
            module = ModuleOp([o])
        else:
            module = o

        stream = StringIO()
        printer = AssemblyPrinter(stream=stream)
        printer.print_module(module)
        assembly = ".intel_syntax\n" + stream.getvalue()
        command = f"llvm-mca --march={self.arch} -mcpu={self.microarch}"
        out,err = run_command(
            command = command,
            input = assembly
        )
        assert err == ''
        throughput = None
        for line in out.split('\n'):
            if line.startswith('Block RThroughput:'):
                throughput = float(line.replace('Block RThroughput:',''))
        assert not throughput is None
        return Cost(throughput)
