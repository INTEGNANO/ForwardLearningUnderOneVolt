import serial
import serial.tools.list_ports

from enum import IntEnum, IntFlag
from enum import auto as en_auto
from functools import reduce
from operator import or_

###################
# C enums and flags


# Shift Registers
class SR(IntEnum):
    HWL = 0
    VWL = en_auto()
    HSL = en_auto()
    VSL = en_auto()
    BL = en_auto()
    ANA = en_auto()


SR_WORD_SIZE = 128
SR_LIST = list(SR.__members__.values())
SR_COUNT = len(SR_LIST)


# Shift Register state code
class State(IntEnum):
    SET = 0x01
    RESET = 0x00

    def __eq__(self, other):
        if other == b"\x01" or other == True:
            return int(self) == 1
        elif other == b"\x00" or other == False:
            return int(self) == 0

        raise ValueError(
            f"Invalid comparaison between a state and an unknown value '{other}'"
        )


# Command list
class CMD(IntEnum):
    SET_SR = 0
    FILL_SRs = en_auto()
    SET_CS = en_auto()
    SET_CS_EN = en_auto()
    SET_SWEEP_MASK = en_auto()
    ARRAY_SWEEP = en_auto()
    GET_CTL = en_auto()

    CLK = en_auto()
    ACK_MODE = en_auto()

    DEBUG_ECHO = en_auto()
    DEBUG_LED = en_auto()


CMD_LIST = list(CMD.__members__.values())
CMD_COUNT = len(CMD_LIST)


# Acknowledge Mode Flags
class ACK(IntFlag):
    NONE = 0x00
    SET_SR = en_auto()
    FILL_SRs = en_auto()
    SET_CS = en_auto()
    SET_CS_EN = en_auto()
    SET_SWEEP_MASK = en_auto()
    ARRAY_SWEEP = en_auto()
    CLK = en_auto()


ACK_LIST = list(ACK.__members__.values())
ACK_ALL = reduce(or_, ACK_LIST)


# Control Signals
class CS(IntEnum):
    EN_HWL = 0
    EN_VWL = en_auto()
    EN_HSL = en_auto()
    EN_VSL = en_auto()
    EN_BL = en_auto()
    EN_INF = en_auto()
    WL_DIG_0 = en_auto()
    WL_DIG_1 = en_auto()
    HSL_DIG_0 = en_auto()
    HSL_DIG_1 = en_auto()
    VSL_DIG_0 = en_auto()
    VSL_DIG_1 = en_auto()
    BL_DIG_0 = en_auto()
    BL_DIG_1 = en_auto()
    BL_DIG_2 = en_auto()


CS_LIST = list(CS.__members__.values())
CS_COUNT = len(CS_LIST)


# Sweep Mask
class SetSweepMask(IntEnum):
    SINGLE = 0
    COL = en_auto()
    ROW = en_auto()
    SET = en_auto()
    WIDTH = en_auto()
    HEIGHT = en_auto()
    COL_OFFSET = en_auto()
    ROW_OFFSET = en_auto()


#################
# Utils function
def as_int(b: bytes) -> int:
    return int.from_bytes(b, "little")


def as_bytes(i: int) -> bytes:
    return i.to_bytes(max((i.bit_length() + 7) // 8, 1), "little")


#################
# Driver class
class MCDriver:
    """
    µc driver for the Awesome Array Python Driver.


    ...
    Attributes
    -----------
    ser : serial.Serial
            serial port associated with the µc

    uc_ack_mode : ACK
            stores the actual ack_mode of the µc

    """

    DEFAULT_PID = 22336

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)

        def gen_command_fn(command):
            return lambda *c_args: self.call_command(command, *c_args)

        for cmd in CMD.__members__:
            setattr(self, cmd.lower(), gen_command_fn(CMD.__members__[cmd]))

        return self

    def __init__(
        self,
        array_size=(32, 32),
        array_offset=(0, 0),
        mask_to_ones=True,
        pid=DEFAULT_PID,
    ):
        """
        Creates the driver.

        Details:
                It will search for the µc using the PID value 'DEFAULT_PID' or the one provided in argument.
                Takes the first found if many have the same PID.
                RAISE if not found.

        Arguments:
                pid: optional, the pid to search for.
                array_size: (int, int), the size of the array in (height, width) format.
                array_offset: (int, int), the offset of the array in (row_offset, col_offset) format.
        """
        self.ser = serial.Serial()
        self.ser.baudrate = 921600
        self.uc_ack_mode = ACK.NONE
        self.array_offset = array_offset
        self.array_size = array_size

        ports = serial.tools.list_ports.comports()
        st_port = None

        for port in ports:
            # If there are multiple serial ports with the same PID, we just use the first one
            if port.pid == pid:
                st_port = port.device
                break

        if st_port is None:
            raise Exception(
                "µc not found, please verify its connection or specify its PID"
            )

        self.ser.port = st_port
        self.ser.open()

        self.reset_state()
        self.set_sweep_mask(SetSweepMask.HEIGHT, array_size[0])
        self.set_sweep_mask(SetSweepMask.WIDTH, array_size[1])
        self.set_sweep_mask(SetSweepMask.ROW_OFFSET, array_offset[0])
        self.set_sweep_mask(SetSweepMask.COL_OFFSET, array_offset[1])
        for i in range(array_size[0]):
            if mask_to_ones:
                self.set_sweep_mask(
                    SetSweepMask.ROW, i, *([0b11111111] * (array_size[1] // 4))
                )
            else:
                self.set_sweep_mask(SetSweepMask.ROW, i, *([0] * (array_size[1] // 4)))

    def __del__(self):
        """Closes the serial port if still open."""
        self.ser.close()

    @staticmethod
    def list_ports():
        """Returns a list of all the serial ports recognized by the OS."""
        return serial.tools.list_ports.comports()

    @staticmethod
    def print_ports():
        """Prints out the useful info about the serial ports recognized by the OS."""
        ports = serial.tools.list_ports.comports()
        if len(ports) == 0:
            print("❌ No serial ports found")
        else:
            for port in serial.tools.list_ports.comports():
                print(port, "| PID: ", port.pid)

    def send_command(self, command, *args, wait_for_ack=False):
        """
        Sends a command to the µc with the optionnaly provided arguments.

        Parameters:
                command: The command to send (see CMD_LIST)
                *args: The provided arguments, which will be converted to bytes

        Returns:
                The actual number of bytes sent.
        """
        if not self.ser.is_open:
            raise Exception("Serial port not open")

        if command == CMD.ACK_MODE:
            self.uc_ack_mode = ACK(args[0])

        command_bytes = as_bytes(command)
        bytes_sent_count = 0

        split_args = []
        max_bytes = 64 - 3  # 64 bytes - '0xAA' - 'command-bytes' - '0xAA'/'0xAB'
        while len(args) > max_bytes:
            split_args.append(args[:max_bytes])
            args = args[max_bytes:]

            # After first loop:
            max_bytes = 64 - 1  # 64 bytes - '0xAA'/'0xAB'

        split_args.append(args)

        # Words sent for instance:
        #  0xAA, CMD, data0 ... data60, 0xAB
        #  data61 .. data123, 0xAB
        #  data124 .. data130, 0xAA
        for i, args in enumerate(split_args):
            cmd = b"\xaa" + command_bytes if i == 0 else b""
            for arg in args:
                if isinstance(arg, int):
                    cmd += as_bytes(arg)
                elif isinstance(arg, list):
                    raise ValueError("Please unpack lists in argument: '*[...]'")
                else:
                    cmd += bytes(arg)
            cmd += b"\xaa" if i + 1 == len(split_args) else b"\xab"

            bytes_sent_count += self.ser.write(cmd)

        if wait_for_ack:
            ack = self.read(2, flush_rest=False)
            if ack != b"\xaa" + command_bytes:
                raise Exception(f"Expected ack for command '{command}', got '{ack}'")

        return bytes_sent_count

    def read(self, size=None, wait_for=True, flush_rest=True):
        """
        Reads from the µc.

        Parameters:
                size:       The number of bytes to read. If None, reads everything.
                wait_for:   When size=None, if True, waits for input ; When size!=None, wait_for is True
                flush_rest: If True, flushes the input buffer if non-empty after {size} bytes have been read.

        Returns:
                The bytes read.
        """
        if not self.ser.is_open:
            raise Exception("Serial port not open")

        if size is None:
            out = b""

            # Block or not until something is in
            while wait_for and not self.ser.in_waiting:
                pass

            # Read everything until the buffer is empty
            while self.ser.in_waiting:
                out += self.ser.read()
            return out

        # else
        out = self.ser.read(size)
        if flush_rest:
            self.flush_input()
        return out

    def call_command(self, command, *args):
        """
        Send a command and waits for a return value if needed

        Parameters:
                command: The command to send (see CMD_LIST)
                *args: The provided arguments, which will be converted to bytes

        Returns:
                The return value of the corresponding command or None

        Details:
                Will expect an ack if set with the corresponding 'action' command (driver.ack_mode(ACK.XXX)) (RAISE if expected and not received)
                Will return the value received if the command is a 'get' command
        """

        wait_for_ack = command.name in self.uc_ack_mode.name
        cmd_returns = (
            (command.name not in ACK_ALL.name) and command != CMD.ACK_MODE
        )  # If the commands does not have an associated ack, it is because it returns something

        if command == CMD.ACK_MODE and args[0] != ACK.NONE:
            wait_for_ack = True
            cmd_returns = False

        self.send_command(command, *args, wait_for_ack=wait_for_ack)

        return self.read() if cmd_returns else None

    def flush_input(self):
        """Flushes the input buffer."""
        while self.ser.in_waiting:
            self.ser.read()

    ### HIGHER LEVELS FUNCTION ###
    def reset_state(self):
        """
        Resets the state of the driver, to run after exception catching for example.
        """
        self.flush_input()  # Flush any remaning inputs stuck in the buffer
        self.ack_mode(ACK.FILL_SRs | ACK.SET_CS_EN | ACK.SET_SWEEP_MASK)
        self.set_cs_en(0b11010)  # Activate BLs and Vertical WLs/SLs

    def get_sr_configuration(self, col: int, row: int):
        """
        Gives the shift registers words for the given memristor location and state.

        Parameters:
                col: int, row: int : Memristor location

        Returns:
                An array containing the computed shift registers words
        """
        # Indices are the same as SR.HWL/HSL/...
        sr_words = [
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ]
        sr_words[SR.VWL] = (0, 1 << (row * 2))
        sr_words[SR.VSL] = ((col >= 32) << ((col * 2) % 64), (col < 32) << (col * 2))
        sr_words[SR.BL] = ((col >= 32) << ((col * 2) % 64), (col < 32) << (col * 2))

        return sr_words

    def configure_sr_words(self, sr_words: list[int]):
        """
        Configure the shift registers with the given words.

        Parameters:
                sr_words: List[int] : The 128bits words to configure the shift registers with.
        """
        args = []
        for word in sr_words:
            # Long live to Python native support for BigInt
            args.extend(word[1].to_bytes(8, "little"))
            args.extend(word[0].to_bytes(8, "little"))

        self.fill_srs(*args)

    def configure_sr(self, col: int, row: int, **kwargs):
        """
        Configure the shift registers for the given memristor location and state.

        Parameters:
                col: int, row: int : Memristor location
                **kwargs : Extra dummy arguments for compatibility with uc_aa

        Returns:
                An array containing the shift registers words set.
        """
        sr_words = self.get_sr_configuration(col, row)
        self.configure_sr_words(sr_words)

        return sr_words

    def test_sr_words_sanity(self, sr_words: list[int]):
        """
        Tests the shift registers sanity for the given words.

        Parameters:
                sr_words: List[int] : The words to configure the shift registers with.

        Returns:
                A 2D array containing the sanity of the shift register's bits.
                Details:
                        sanity[shitreg_id][bit_id] = True if sane, False otherwise
        """
        self.configure_sr_words(sr_words)

        sanity = [[None for _ in range(SR_WORD_SIZE)] for _ in range(SR_COUNT)]

        for bit_id in reversed(range(SR_WORD_SIZE)):
            for sr_id, sr_word in enumerate(sr_words):
                set_val = ((sr_word >> bit_id) & 1) == 1
                sr_val = self.get_ctl(sr_id) == State.SET
                sanity[sr_id][bit_id] = set_val == sr_val

            self.clk()  # Shift once

        return sanity

    def flush_sr(self):
        content = [[None for _ in range(SR_WORD_SIZE)] for _ in range(SR_COUNT)]

        for bit_id in reversed(range(SR_WORD_SIZE)):
            for sr_id in range(SR_COUNT):
                sr_val = self.get_ctl(sr_id) == State.SET

                content[sr_id][bit_id] = sr_val

            self.clk()  # Shift once

        return content

    def test_sr_sanity(self, col: int, row: int, bar: bool, set: bool):
        """
        Tests the shift registers sanity for the given configuration of memristor location and state.

        Parameters:
                col: int, row: int, bar: bool : Memristor location
                set: bool : Memristor state

        Returns:
                A 2D array containing the sanity of the shift register's bits.
                Details:
                        sanity[shitreg_id][bit_id] = True if sane, False otherwise
        """
        sr_words = self.get_sr_configuration(col, row, bar, set)
        return self.test_sr_words_sanity(sr_words)
