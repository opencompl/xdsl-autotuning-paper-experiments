"""Run a generated AMX GEMM kernel without AMX hardware.

The kernel is linked into a shared object and disassembled with `objdump`;
this module then interprets the disassembly instruction by instruction, with
just enough of x86-64 for the code the GEMM generators emit: moves and integer
arithmetic on general-purpose registers, the stack, compares and conditional
jumps, the AMX tile instructions, and the AVX-512 moves and conversions that
carry a BF16 C between memory and the F32 tiles. Tile instructions follow their
pseudocode in the Intel SDM, including the shape checks that make a mismatched
tile fault, so a kernel whose tile configuration does not fit its tiles fails
here as it would on a Sapphire Rapids; masked vector moves touch only the
elements their mask selects, as the hardware's fault suppression does.

Anything else -- an instruction or operand form it does not know -- raises, so
a kernel either runs completely or is reported as unsupported.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

_GPR64 = tuple(
    f"r{name}" for name in ("ax", "cx", "dx", "bx", "sp", "bp", "si", "di")
) + tuple(f"r{index}" for index in range(8, 16))
_GPR32 = tuple(
    f"e{name}" for name in ("ax", "cx", "dx", "bx", "sp", "bp", "si", "di")
) + tuple(f"r{index}d" for index in range(8, 16))
_MASK64 = (1 << 64) - 1
_SIZES = {"BYTE": 1, "WORD": 2, "DWORD": 4, "QWORD": 8}
_VECTOR_SIZES = {"XMMWORD": 16, "YMMWORD": 32, "ZMMWORD": 64}
_VECTOR_WIDTHS = {"x": 16, "y": 32, "z": 64}
# The bytes per element each masked vector move uses its mask for.
_MOVE_ELEMENTS = {"vmovups": 4, "vmovdqu16": 2}
# A sentinel return address: `ret` to it ends the call.
_RETURN_ADDRESS = 0xDEAD0000


class EmulationError(Exception):
    """The kernel did something the emulator cannot or would not do."""


@dataclass(frozen=True)
class Instruction:
    address: int
    next_address: int
    mnemonic: str
    operands: tuple[str, ...]
    # The absolute address objdump resolves a rip-relative operand to.
    target: int | None


@dataclass
class Tile:
    rows: int
    colsb: int
    data: npt.NDArray[np.uint8] = field(
        default_factory=lambda: np.zeros((16, 64), np.uint8)
    )


class Memory:
    """Byte-addressed memory made of disjoint regions."""

    def __init__(self) -> None:
        self.regions: list[tuple[int, bytearray]] = []

    def map(self, start: int, data: bytes | bytearray) -> None:
        end = start + len(data)
        for other_start, other in self.regions:
            if start < other_start + len(other) and other_start < end:
                raise EmulationError(f"region at {start:#x} overlaps {other_start:#x}")
        self.regions.append((start, bytearray(data)))

    def _locate(self, address: int, size: int) -> tuple[bytearray, int]:
        for start, data in self.regions:
            if start <= address and address + size <= start + len(data):
                return data, address - start
        raise EmulationError(f"access of {size} bytes at {address:#x} is unmapped")

    def read(self, address: int, size: int) -> bytes:
        data, offset = self._locate(address, size)
        return bytes(data[offset : offset + size])

    def write(self, address: int, value: bytes) -> None:
        data, offset = self._locate(address, len(value))
        data[offset : offset + len(value)] = value


def _run(*command: str) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def _load_sections(library: Path, memory: Memory) -> None:
    """Map the bytes of every allocated section `objdump -s` dumps."""
    section: int | None = None
    chunks: dict[int, bytearray] = {}
    for line in _run("objdump", "-s", str(library)).splitlines():
        if line.startswith("Contents of section"):
            section = None
            continue
        match = re.match(r"^ ([0-9a-f]+) ((?:[0-9a-f]{2,8} ?){1,4})", line)
        if not match:
            continue
        address = int(match.group(1), 16)
        data = bytes.fromhex(match.group(2).replace(" ", ""))
        if section is not None and address == section + len(chunks[section]):
            chunks[section] += data
        else:
            section = address
            chunks[section] = bytearray(data)
    for start, data in chunks.items():
        memory.map(start, data)


def _disassemble(library: Path, symbol: str) -> dict[int, Instruction]:
    instructions: dict[int, Instruction] = {}
    in_symbol = False
    lines = _run(
        "objdump", "-d", "-M", "intel", "--no-show-raw-insn", str(library)
    ).splitlines()
    parsed: list[tuple[int, str]] = []
    for line in lines:
        # Labels inside the kernel show up as symbols of their own, so the
        # kernel runs from its symbol to the end of its section.
        if line.startswith("Disassembly of section"):
            in_symbol = False
            continue
        if header := re.match(r"^[0-9a-f]+ <(.+)>:$", line):
            in_symbol = in_symbol or header.group(1) == symbol
            continue
        if not in_symbol:
            continue
        match = re.match(r"^\s+([0-9a-f]+):\s+(.*)$", line)
        if match and match.group(2).strip() and not match.group(2).startswith("("):
            parsed.append((int(match.group(1), 16), match.group(2).strip()))
    if not parsed:
        raise EmulationError(f"no instructions for {symbol}")
    for (address, text), following in zip(
        parsed, [*parsed[1:], (None, "")], strict=True
    ):
        comment = re.search(r"#\s*([0-9a-f]+)", text)
        text = text.split("#")[0].strip()
        jump = re.match(r"^(j\w+)\s+([0-9a-f]+) <", text)
        if jump:
            mnemonic, operands = jump.group(1), (jump.group(2),)
            target = int(jump.group(2), 16)
        else:
            mnemonic, _, rest = text.partition(" ")
            operands = tuple(_split_operands(rest.strip()))
            target = int(comment.group(1), 16) if comment else None
        next_address = following[0] if following[0] is not None else address + 16
        instructions[address] = Instruction(
            address, next_address, mnemonic, operands, target
        )
    return instructions


def _split_operands(text: str) -> list[str]:
    operands: list[str] = []
    depth = 0
    current = ""
    for char in text:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == "," and depth == 0:
            operands.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        operands.append(current.strip())
    return operands


def _bf16_to_f32(words: npt.NDArray[np.uint16]) -> npt.NDArray[np.float32]:
    return (words.astype(np.uint32) << 16).view(np.float32)


def convert_fp32_to_bfloat16(values: npt.NDArray[np.float32]) -> npt.NDArray[np.uint16]:
    """The SDM's pseudo-function of vcvtne(2)ps2bf16: round to nearest even,
    denormals to zeros of their sign, NaNs quieted."""
    bits = values.view(np.uint32).astype(np.uint64)
    exponent = (bits >> 23) & 0xFF
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    result = np.where(exponent == 0, (bits >> 16) & 0x8000, rounded)
    nan = (exponent == 0xFF) & ((bits & 0x7FFFFF) != 0)
    result = np.where(nan, (bits >> 16) | 0x40, result)
    infinity = (exponent == 0xFF) & ((bits & 0x7FFFFF) == 0)
    result = np.where(infinity, bits >> 16, result)
    return result.astype(np.uint16)


class Emulator:
    def __init__(self, library: Path, symbol: str) -> None:
        self.memory = Memory()
        _load_sections(library, self.memory)
        self.instructions = _disassemble(library, symbol)
        self.entry = min(self.instructions)
        self.registers = dict.fromkeys(_GPR64, 0)
        self.masks: dict[str, int] = {}
        self.flags: tuple[int, int] | None = None
        self.palette = 0
        self.tiles = [Tile(0, 0) for _ in range(8)]
        self.vectors = np.zeros((32, 64), np.uint8)
        self.executed = 0
        self.mnemonics: dict[str, int] = {}

    # -- operands ---------------------------------------------------------

    def _register(self, name: str) -> int:
        if name in _GPR64:
            return self.registers[name]
        if name in _GPR32:
            return self.registers[_GPR64[_GPR32.index(name)]] & 0xFFFFFFFF
        raise EmulationError(f"unknown register {name}")

    def _set_register(self, name: str, value: int) -> None:
        if name in _GPR64:
            self.registers[name] = value & _MASK64
        elif name in _GPR32:
            self.registers[_GPR64[_GPR32.index(name)]] = value & 0xFFFFFFFF
        else:
            raise EmulationError(f"unknown register {name}")

    def _address(self, operand: str, instruction: Instruction) -> int:
        expression = operand[operand.index("[") + 1 : operand.index("]")]
        if expression.startswith("rip"):
            if instruction.target is None:
                raise EmulationError(f"unresolved {operand}")
            return instruction.target
        address = 0
        for sign, term in re.findall(r"([+-]?)([^+-]+)", expression):
            if "*" in term:
                register, scale = term.split("*")
                value = self._register(register) * int(scale)
            elif term in _GPR64 or term in _GPR32:
                value = self._register(term)
            else:
                value = int(term, 0)
            address += -value if sign == "-" else value
        return address & _MASK64

    @staticmethod
    def _size(operand: str) -> int:
        word = operand.split()[0]
        if word not in _SIZES:
            raise EmulationError(f"memory operand without a size: {operand}")
        return _SIZES[word]

    def _value(self, operand: str, instruction: Instruction) -> int:
        if "[" in operand:
            size = self._size(operand)
            data = self.memory.read(self._address(operand, instruction), size)
            return int.from_bytes(data, "little")
        if operand in _GPR64 or operand in _GPR32:
            return self._register(operand)
        return int(operand, 0) & _MASK64

    @staticmethod
    def _signed(value: int) -> int:
        value &= _MASK64
        return value - (1 << 64) if value >> 63 else value

    # -- vectors ----------------------------------------------------------

    @staticmethod
    def _vector_register(operand: str) -> tuple[int, int, str | None, bool]:
        """The index and byte width of a vector register operand, with its
        write mask and whether that zeroes, e.g. `ymm3{k1}{z}`."""
        match = re.fullmatch(r"([xyz])mm(\d+)((?:\{\w+\})*)", operand)
        if match is None:
            raise EmulationError(f"not a vector register: {operand}")
        decorations = re.findall(r"\{(\w+)\}", match.group(3))
        mask = next((d for d in decorations if d.startswith("k")), None)
        return (
            int(match.group(2)),
            _VECTOR_WIDTHS[match.group(1)],
            mask,
            "z" in decorations,
        )

    def _mask_bits(self, mask: str | None, elements: int) -> list[bool]:
        if mask is None:
            return [True] * elements
        if mask not in self.masks:
            raise EmulationError(f"{mask} is not set")
        return [bool(self.masks[mask] >> element & 1) for element in range(elements)]

    def _read_vector(self, operand: str, instruction: Instruction) -> bytes:
        """A whole vector source operand, register or memory."""
        if "[" in operand:
            word = operand.split()[0]
            if word not in _VECTOR_SIZES or re.search(r"\{k\d\}", operand):
                raise EmulationError(f"unsupported vector memory operand: {operand}")
            return self.memory.read(
                self._address(operand, instruction), _VECTOR_SIZES[word]
            )
        index, width, mask, _ = self._vector_register(operand)
        if mask is not None:
            raise EmulationError(f"masked vector source: {operand}")
        return self.vectors[index, :width].tobytes()

    def _write_vector(self, operand: str, data: bytes) -> None:
        """Write an unmasked vector result, zeroing the register above it."""
        index, width, mask, _ = self._vector_register(operand)
        if mask is not None or len(data) != width:
            raise EmulationError(f"unsupported vector destination: {operand}")
        self.vectors[index] = 0
        self.vectors[index, :width] = np.frombuffer(data, np.uint8)

    def _vector_move(
        self, destination: str, source: str, element: int, instruction: Instruction
    ) -> None:
        """vmovups or vmovdqu16, either way, masked or not."""
        if "[" in destination:
            index, width, _, _ = self._vector_register(source)
            mask = re.search(r"\{(k\d)\}", destination)
            address = self._address(destination, instruction)
            data = self.vectors[index, :width].tobytes()
            selected = self._mask_bits(mask and mask.group(1), width // element)
            for position, write in enumerate(selected):
                if write:
                    start = position * element
                    self.memory.write(address + start, data[start : start + element])
            return
        index, width, mask, zeroing = self._vector_register(destination)
        if "[" in source:
            address = self._address(source, instruction)
            result = bytearray(self.vectors[index, :width].tobytes())
            selected = self._mask_bits(mask, width // element)
            for position, read in enumerate(selected):
                start = position * element
                if read:
                    result[start : start + element] = self.memory.read(
                        address + start, element
                    )
                elif zeroing or mask is None:
                    result[start : start + element] = bytes(element)
            if mask is not None and not zeroing:
                raise EmulationError(f"merge-masked load: {destination}")
            self.vectors[index] = 0
            self.vectors[index, :width] = np.frombuffer(bytes(result), np.uint8)
            return
        if mask is not None:
            raise EmulationError(f"masked register move: {destination}")
        self._write_vector(destination, self._read_vector(source, instruction))

    # -- tiles ------------------------------------------------------------

    def _tile(self, name: str) -> Tile:
        if not name.startswith("tmm"):
            raise EmulationError(f"not a tile: {name}")
        tile = self.tiles[int(name[3:])]
        if self.palette == 0 or tile.rows == 0:
            raise EmulationError(f"{name} is not configured")
        return tile

    def _load_tile_config(self, address: int) -> None:
        config = self.memory.read(address, 64)
        self.palette = config[0]
        if self.palette not in (0, 1):
            raise EmulationError(f"unsupported palette {self.palette}")
        if any(config[1:16]) or any(config[32:48]) or any(config[56:]):
            raise EmulationError("reserved tile configuration bytes are set")
        for index, tile in enumerate(self.tiles):
            tile.colsb = int.from_bytes(
                config[16 + 2 * index : 18 + 2 * index], "little"
            )
            tile.rows = config[48 + index]
            if (
                tile.rows > 16
                or tile.colsb > 64
                or (tile.rows == 0) != (tile.colsb == 0)
            ):
                raise EmulationError(f"invalid shape for tmm{index}")
            tile.data[:] = 0

    def _store_tile_config(self, address: int) -> None:
        config = bytearray(64)
        if self.palette:
            config[0] = self.palette
            for index, tile in enumerate(self.tiles):
                config[16 + 2 * index : 18 + 2 * index] = tile.colsb.to_bytes(
                    2, "little"
                )
                config[48 + index] = tile.rows
        self.memory.write(address, bytes(config))

    def _tile_rows(self, operand: str, instruction: Instruction) -> tuple[int, int]:
        """The address of a tile memory operand's first row, and the stride."""
        expression = operand[operand.index("[") + 1 : operand.index("]")]
        index = re.search(r"(\w+)\*(\d)", expression)
        if index is None:
            raise EmulationError(f"tile memory operand without an index: {operand}")
        stride = self._register(index.group(1)) * int(index.group(2))
        base = self._address(operand.replace(index.group(0), "0"), instruction)
        return base, stride

    def _tdpbf16ps(self, destination: Tile, a: Tile, b: Tile) -> None:
        # SDM: dst.rows == a.rows, b.colsb == dst.colsb, a.colsb / 4 == b.rows.
        if a.rows != destination.rows or b.colsb != destination.colsb:
            raise EmulationError("tdpbf16ps: tile shapes do not match")
        if a.colsb // 4 != b.rows or a.colsb % 4 or destination.colsb % 4:
            raise EmulationError("tdpbf16ps: tile K extents do not match")
        rows, pairs, columns = destination.rows, b.rows, destination.colsb // 4
        a_words = a.data[:rows, : 4 * pairs].copy().view(np.uint16)
        b_words = b.data[:pairs, : 4 * columns].copy().view(np.uint16)
        a_values = _bf16_to_f32(a_words).reshape(rows, pairs, 2).astype(np.float64)
        b_values = _bf16_to_f32(b_words).reshape(pairs, columns, 2).astype(np.float64)
        product = np.einsum("mkp,knp->mn", a_values, b_values)
        accumulator = destination.data[:rows, : 4 * columns].copy().view(np.float32)
        result = (accumulator.astype(np.float64) + product).astype(np.float32)
        destination.data[:rows, : 4 * columns] = result.view(np.uint8)

    # -- execution --------------------------------------------------------

    def call(self, arguments: tuple[int, ...], stack_size: int = 1 << 20) -> None:
        """Call the kernel with up to six integer arguments."""
        stack_top = 0x7F0000000000
        self.memory.map(stack_top - stack_size, bytes(stack_size))
        for register, value in zip(("rdi", "rsi", "rdx", "rcx", "r8", "r9"), arguments):
            self.registers[register] = value
        self.registers["rsp"] = stack_top - 64
        self._push(_RETURN_ADDRESS)
        address = self.entry
        while address != _RETURN_ADDRESS:
            if address not in self.instructions:
                raise EmulationError(f"jump to {address:#x}, outside the kernel")
            instruction = self.instructions[address]
            address = self._execute(instruction)
            self.executed += 1
            self.mnemonics[instruction.mnemonic] = (
                self.mnemonics.get(instruction.mnemonic, 0) + 1
            )
            if self.executed > 50_000_000:
                raise EmulationError("too many instructions; is a loop unbounded?")

    def _push(self, value: int) -> None:
        self.registers["rsp"] = (self.registers["rsp"] - 8) & _MASK64
        self.memory.write(self.registers["rsp"], value.to_bytes(8, "little"))

    def _pop(self) -> int:
        value = int.from_bytes(self.memory.read(self.registers["rsp"], 8), "little")
        self.registers["rsp"] = (self.registers["rsp"] + 8) & _MASK64
        return value

    def _execute(self, instruction: Instruction) -> int:
        mnemonic, operands = instruction.mnemonic, instruction.operands
        following = instruction.next_address
        match mnemonic:
            case "nop" | "endbr64" | "vzeroupper" | "xchg" if mnemonic != "xchg" or (
                operands == ("ax", "ax")
            ):
                pass
            case "push":
                self._push(self._value(operands[0], instruction))
            case "pop":
                self._set_register(operands[0], self._pop())
            case "ret":
                return self._pop()
            case "mov" | "movabs":
                destination, source = operands
                value = self._value(source, instruction)
                if "[" in destination:
                    size = self._size(destination)
                    self.memory.write(
                        self._address(destination, instruction),
                        (value & ((1 << (8 * size)) - 1)).to_bytes(size, "little"),
                    )
                else:
                    self._set_register(destination, value)
            case "lea":
                self._set_register(operands[0], self._address(operands[1], instruction))
            case "add" | "sub" | "and" | "or" | "xor" | "imul":
                destination, source = operands
                if "[" in destination:
                    raise EmulationError(f"{mnemonic} to memory")
                left = self._register(destination)
                right = self._value(source, instruction)
                result = {
                    "add": left + right,
                    "sub": left - right,
                    "and": left & right,
                    "or": left | right,
                    "xor": left ^ right,
                    "imul": self._signed(left) * self._signed(right),
                }[mnemonic]
                self._set_register(destination, result)
                self.flags = (self._signed(result), 0)
            case "cmp":
                self.flags = (
                    self._signed(self._value(operands[0], instruction)),
                    self._signed(self._value(operands[1], instruction)),
                )
            case "jmp":
                return int(operands[0], 16)
            case "jl" | "jge" | "jle" | "jg" | "jne" | "je" | "jb" | "jae":
                if self.flags is None:
                    raise EmulationError(f"{mnemonic} without a compare")
                left, right = self.flags
                taken = {
                    "jl": left < right,
                    "jge": left >= right,
                    "jle": left <= right,
                    "jg": left > right,
                    "jne": left != right,
                    "je": left == right,
                    "jb": (left & _MASK64) < (right & _MASK64),
                    "jae": (left & _MASK64) >= (right & _MASK64),
                }[mnemonic]
                if taken:
                    return int(operands[0], 16)
            case "kmovw" | "kmovd" | "kmovq" | "kmovb":
                self.masks[operands[0]] = self._value(operands[1], instruction)
            case "vmovups" | "vmovdqu16":
                self._vector_move(
                    operands[0], operands[1], _MOVE_ELEMENTS[mnemonic], instruction
                )
            case "vpmovsxwd":
                words = np.frombuffer(
                    self._read_vector(operands[1], instruction), np.int16
                )
                self._write_vector(operands[0], words.astype(np.int32).tobytes())
            case "vpslld":
                doublewords = np.frombuffer(
                    self._read_vector(operands[1], instruction), np.uint32
                )
                shift = int(operands[2], 0)
                shifted = doublewords << shift if shift < 32 else 0 * doublewords
                self._write_vector(operands[0], shifted.astype(np.uint32).tobytes())
            case "vcvtne2ps2bf16":
                # The second source fills the low half, the first the high one.
                high, low = (
                    np.frombuffer(self._read_vector(operand, instruction), np.float32)
                    for operand in operands[1:]
                )
                self._write_vector(
                    operands[0],
                    convert_fp32_to_bfloat16(np.concatenate((low, high))).tobytes(),
                )
            case "vcvtneps2bf16":
                values = np.frombuffer(
                    self._read_vector(operands[1], instruction), np.float32
                )
                self._write_vector(
                    operands[0], convert_fp32_to_bfloat16(values).tobytes()
                )
            case "ldtilecfg":
                self._load_tile_config(self._address(operands[0], instruction))
            case "sttilecfg":
                self._store_tile_config(self._address(operands[0], instruction))
            case "tilerelease":
                self.palette = 0
                for tile in self.tiles:
                    tile.rows = tile.colsb = 0
            case "tilezero":
                self._tile(operands[0]).data[:] = 0
            case "tileloadd" | "tileloaddt1":
                tile = self._tile(operands[0])
                start, stride = self._tile_rows(operands[1], instruction)
                tile.data[:] = 0
                for row in range(tile.rows):
                    data = self.memory.read(start + row * stride, tile.colsb)
                    tile.data[row, : tile.colsb] = np.frombuffer(data, np.uint8)
            case "tilestored":
                tile = self._tile(operands[1])
                start, stride = self._tile_rows(operands[0], instruction)
                for row in range(tile.rows):
                    self.memory.write(
                        start + row * stride, tile.data[row, : tile.colsb].tobytes()
                    )
            case "tdpbf16ps":
                self._tdpbf16ps(
                    self._tile(operands[0]),
                    self._tile(operands[1]),
                    self._tile(operands[2]),
                )
            case _:
                raise EmulationError(
                    f"unsupported instruction at {instruction.address:#x}: "
                    f"{mnemonic} {', '.join(operands)}"
                )
        return following


def bf16_round(values: npt.NDArray[np.float32]) -> npt.NDArray[np.uint16]:
    """Round F32 values to BF16 bit patterns, to nearest even."""
    bits = values.astype(np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def pack_vnni_a(a: npt.NDArray[np.uint16], m: int, k: int) -> npt.NDArray[np.uint16]:
    """Pack column-major M-by-K A into LIBXSMM's VNNI-2 layout: K pair p of row
    m is at `p * 2 * m_count + 2 * m`, as `headers/vnni.h` does."""
    columns = a.reshape(k, m)
    packed = np.zeros(k * m, np.uint16)
    for kk in range(k):
        packed[(kk // 2) * 2 * m + 2 * np.arange(m) + kk % 2] = columns[kk]
    return packed


def run_gemm(
    library: Path, m: int, n: int, k: int, seed: int = 0, *, bf16_c: bool = False
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64], Emulator]:
    """Run `matmul(A, B, C)` from `library`, a BF16 GEMM with A in VNNI-2
    layout, on random data; return C and a reference for it.

    With `bf16_c`, C is BF16 too: the kernel accumulates in F32 and rounds the
    result back, so the returned C is that BF16 widened to F32, and the reference
    is computed from the BF16 C it starts with."""
    rng = np.random.default_rng(seed)
    a = bf16_round(rng.uniform(-1, 1, m * k).astype(np.float32))
    b = bf16_round(rng.uniform(-1, 1, k * n).astype(np.float32))
    c = rng.uniform(-1, 1, m * n).astype(np.float32)
    c_bytes = c.tobytes()
    if bf16_c:
        c_words = bf16_round(c)
        c, c_bytes = _bf16_to_f32(c_words), c_words.tobytes()
    reference = (
        _bf16_to_f32(a).astype(np.float64).reshape(k, m).T
        @ _bf16_to_f32(b).astype(np.float64).reshape(n, k).T
        + c.astype(np.float64).reshape(n, m).T
    )
    emulator = Emulator(library, "matmul")
    # Guard pages around every matrix catch out-of-bounds accesses.
    a_address, b_address, c_address = 0x10000000, 0x20000000, 0x30000000
    emulator.memory.map(a_address, pack_vnni_a(a, m, k).tobytes())
    emulator.memory.map(b_address, b.tobytes())
    emulator.memory.map(c_address, c_bytes)
    emulator.call((a_address, b_address, c_address))
    raw = emulator.memory.read(c_address, len(c_bytes))
    result = (
        _bf16_to_f32(np.frombuffer(raw, np.uint16))
        if bf16_c
        else np.frombuffer(raw, np.float32)
    )
    return result.reshape(n, m).T.copy(), reference, emulator
