import SoapySDR
import sys
import argparse
import math
import numpy as np
import bitstring
import binascii
import crcmod

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--bandwidth', type=int, default=5e6)
parser.add_argument('--rf', type=int, default=2420e6)
parser.add_argument('--tx-gain', type=float, default=50.0)
parser.add_argument('--tx-antenna', type=str, default='BAND1')
parser.add_argument('--rx-gain', type=float, default=30.0)
parser.add_argument('--rx-antenna', type=str, default='LNAW')
parser.add_argument('--my-address', type=int, default='1')
parser.add_argument('--remote-address', type=int, default='2')

# Constant values
PREAMBLE = bitstring.BitArray(hex='aa') * 16
SFD = bitstring.BitArray(hex='2bd4')
MODULATION_BPSK = bitstring.BitArray(hex='01')
BYTE_ZERO = bitstring.BitArray(hex='00')
FRAME_TYPE_DATA = bitstring.BitArray(hex='00')

# Definitions of protocol parameters
SAMPLE_RATE = 1e6
SAMPLES_PER_SYMBOL = 10
RECEIVE_BUFFER_SIZE = 100 * SAMPLES_PER_SYMBOL
RECEIVE_SIGNAL_THRESHOLD = 0.02

"""
PHysical & data link layer protocol class
"""
class MyProtocol:
    """
    Constructor
    """
    def __init__(self):
        self.modulation = None
        self.length = 0
    """
    Parse symbols to get a frame
    """
    def parse(self, data):
        # Physical layer
        self.modulation = data[0:8].int
        self.length = data[16:32].int
        # Check the checksum∫
        if not crc16_check(data[0:48].bytes):
            return False
        # Data-link layer
        self.frame_type = data[48:56].int
        self.symbols = data[56:72].int
        self.destination = data[72:104].int
        self.source = data[104:136].int
        self.sequence_number = data[136:152].int
        if data.len - 48 < self.symbols:
            return False
        self.payload = data[152:16 + self.symbols].bytes
        # Check the checksum
        if not crc32_check(data[48:48 + self.symbols].bytes):
            return False
        return True

"""
SdrInterface
"""
class SdrInterface:
    """
    Constructor
    """
    def __init__(self, sdr, address):
        self.sdr = sdr
        self.address = address
        self.connections = {}
        # Initialize the tx/rx streams
        self.txStream = sdr.setupStream(SoapySDR.SOAPY_SDR_TX, SoapySDR.SOAPY_SDR_CF32, [0])
        self.rxStream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [0])
        # Activate the streams
        sdr.activateStream(self.txStream)
        sdr.activateStream(self.rxStream)

    """
    Create a new connection
    """
    def newConnection(self, target):
        if target in self.connections:
            # Connection already exists
            return False
        conn = SdrConnection(self, target)
        self.connections[target] = conn
        return conn

    """
    Transmit
    """
    def xmit(self, dst, src, seqno, data):
        # Postamble
        postamble = np.ones(128, dtype=np.complex64)
        # Build the datalink layer frame
        frame = build_datalink(dst, src, seqno, bitstring.BitArray(data))
        # Build the physical layer protocol header
        phy = build_phy(frame.size)
        # Combine the physical layer header and the data-link frame
        symbols = np.concatenate([phy, frame, postamble])

        # Get samples from symbols
        samples = np.repeat(symbols, SAMPLES_PER_SYMBOL)

        mtu = self.sdr.getStreamMTU(self.txStream)
        sent = 0
        while sent < len(samples):
            chunk = samples[sent:sent+mtu]
            status = self.sdr.writeStream(self.txStream, [chunk], chunk.size, timeoutUs=1000000)
            if status.ret != chunk.size:
                sys.stderr.write("Failed to transmit all samples in writeStream(): {}\n".format(status.ret))
                return False
            sent += status.ret

        return True

"""
SdrConnection
"""
class SdrConnection:
    """
    Constructor
    """
    def __init__(self, iface, target, recvCallback):
        self.iface = iface
        self.target = target
        self.recvCallback = recvCallback
        self.seqno = 0

    """
    Send data
    """
    def send(self, data):
        src = bitstring.BitArray(int=self.iface.address, length=32)
        dst = bitstring.BitArray(int=self.target, length=32)
        self.seqno += 1
        return self.iface.xmit(dst, src, self.seqno, data)
        #return transmit_packet(self.sdr, self.txStream, dst, src, self.seqno, data)


"""
Main routine
"""
def main(args):
    # Create an SDR device instance
    try:
        sdr = SoapySDR.Device(dict(driver="lime"))
    except:
        sys.stderr.write("Failed to create an SDR device instance.\n")
        return False
    if not sdr:
        sys.stderr.write("Could not find any SDR devices.\n")
        return False

    # Setup the Tx channel 0
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_TX, 0, SAMPLE_RATE)
    sdr.setBandwidth(SoapySDR.SOAPY_SDR_TX, 0, args.bandwidth)
    sdr.setAntenna(SoapySDR.SOAPY_SDR_TX, 0, args.tx_antenna)
    sdr.setGain(SoapySDR.SOAPY_SDR_TX, 0, "PAD", args.tx_gain)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, args.rf)

    # Setup the Rx channel 0
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, SAMPLE_RATE)
    sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, args.bandwidth)
    sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, args.rx_antenna)
    sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, "LNA", args.rx_gain)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, args.rf)

    # Prepare a receive buffer
    rxBuffer = np.zeros(RECEIVE_BUFFER_SIZE, np.complex64)

    # Initialize an SDR interface
    iface = SdrInterface(sdr, args.my_address)

    # Initialize a new connection
    conn = iface.newConnection(args.remote_address, None)

    # Loop
    while True:
        # Receive samples
        status = sdr.readStream(rxStream, [rxBuffer], rxBuffer.size)
        if status.ret != rxBuffer.size:
            sys.stderr.write("Failed to receive samples in readStream(): {}\n".format(status.ret))
            return False
        # Detect the edge offset
        samples = rxBuffer
        edgeOffset = detectEdge(samples, SAMPLES_PER_SYMBOL)
        if not edgeOffset:
            # No edge detected
            continue
        if edgeOffset + 4 >= SAMPLES_PER_SYMBOL:
            edgeOffset -= SAMPLES_PER_SYMBOL
        # Decode symbols from the center of a set of samples
        symbols = samples[range(edgeOffset + 4, samples.size, SAMPLES_PER_SYMBOL)]
        # Detect the preamble
        preamblePosition = detectPreamble(symbols)
        if not preamblePosition:
            # Preamble not detected
            continue
        # Receive the samples while the signal is valid
        packetSymbols = np.copy(symbols[preamblePosition:])
        while True:
            status = sdr.readStream(rxStream, [rxBuffer], rxBuffer.size)
            if status.ret != rxBuffer.size:
                sys.stderr.write("Failed to receive samples in readStream(): {}\n".format(status.ret))
                return False
            samples = rxBuffer
            symbols = samples[range(edgeOffset + 4, samples.size, SAMPLES_PER_SYMBOL)]
            packetSymbols = np.concatenate([packetSymbols, symbols])
            if np.sum(np.abs(symbols) > RECEIVE_SIGNAL_THRESHOLD) != symbols.size:
                print("Packet end: # of symbols = {}".format(packetSymbols.size))
                break
        data = demodulate(packetSymbols)

    # Deactivate and close the stream
    sdr.deactivateStream(rxStream)
    sdr.closeStream(rxStream)

    return True


"""
CRC-16
"""
def crc16(bstr):
    f = crcmod.predefined.mkPredefinedCrcFun('crc-aug-ccitt')
    return f(bstr)
def crc16_checksum(bstr):
    return crc16(bstr).to_bytes(2, 'big')
def crc16_check(bstr):
    if crc16(bstr) == 0:
        return True
    else:
        return False

"""
CRC-32
"""
def crc32(bstr):
    f = crcmod.predefined.mkPredefinedCrcFun('crc32')
    return f(bstr)
def crc32_checksum(bstr):
    return crc32(bstr).to_bytes(4, 'little')
def crc32_check(bstr):
    if crc32(bstr) == 0x2144df1c:
        return True
    else:
        return False

"""
Transmit
"""
def transmit_packet(sdr, txStream, dst, src, seqno, data):
    # Postamble
    postamble = np.ones(128, dtype=np.complex64)
    # Build the datalink layer frame
    frame = build_datalink(dst, src, seqno, bitstring.BitArray(data))
    # Build the physical layer protocol header
    phy = build_phy(frame.size)
    # Combine the physical layer header and the data-link frame
    symbols = np.concatenate([phy, frame, postamble])

    # Get samples from symbols
    samples = np.repeat(symbols, SAMPLES_PER_SYMBOL)

    mtu = sdr.getStreamMTU(txStream)
    sent = 0
    while sent < len(samples):
        chunk = samples[sent:sent+mtu]
        status = sdr.writeStream(txStream, [chunk], chunk.size, timeoutUs=1000000)
        if status.ret != chunk.size:
            sys.stderr.write("Failed to transmit all samples in writeStream(): {}\n".format(status.ret))
            return False
        sent += status.ret

    return True

"""
Build phy header
"""
def build_phy(length):
    # Physical layer
    hdr = MODULATION_BPSK + BYTE_ZERO + bitstring.BitArray(int=length, length=16)
    # CRC-16
    cksum = crc16_checksum(hdr.bytes)
    phy = PREAMBLE + SFD + hdr + bitstring.BitArray(bytes=cksum, length=16)
    # Modulate
    return modulate_bpsk(phy)

"""
Build datalink frame
"""
def build_datalink(dst, src, seqno, data):
    # Count the number of symbols (including CRC-32)
    symbols = bitstring.BitArray(int=data.len + 136, length=16)
    # Sequence number
    seq = bitstring.BitArray(int=seqno, length=16)
    # Build a frame
    frame = FRAME_TYPE_DATA + symbols + dst + src + seq + data
    # Calculate the checksum
    cksum = crc32_checksum(frame.bytes)
    frame += cksum
    return modulate_bpsk(frame)

"""
Modulate
"""
def modulate_bpsk(data):
    return np.exp( 1j * math.pi * np.array(data, np.complex64) ).astype(np.complex64)


"""
Detect edge
"""
def detectEdge(samples, samples_per_symbol):
    # Amplitude
    amps = np.abs(samples)
    # Calculate the phase change points
    shifted = samples[1:]
    orig = samples[0:-1]
    orig = np.where(orig==0, orig + 1e-9, orig) # to avoid zero division
    x = shifted / orig
    diffAngles = np.arctan2(x.imag, x.real)
    # Find the change points of a phase (also checking the amplitude)
    changePoints = np.where( (amps[0:-1] > RECEIVE_SIGNAL_THRESHOLD)
        & (np.absolute(diffAngles) > math.pi / 2))[0]
    # Find the edge index
    if changePoints.size == 0:
        return False
    bc = np.bincount(changePoints % samples_per_symbol)
    return np.argmax(bc)

"""
Detect preamble
"""
def detectPreamble(symbols):
    # Amplitude and angles
    amps = np.abs(symbols)
    # Calculate angles from previous symbols
    cur = symbols[1:] # current symbols
    prev = symbols[0:-1] # previous symbols
    prev = np.where(prev==0, prev + 1e-9, prev) # to avoid zero division
    diffAngles = np.angle(cur / prev)
    # Detect part of the preamble using alternating 16 symbols
    pattern = bitstring.BitArray(hex='ffff')
    binary = np.where((amps[0:-1] > RECEIVE_SIGNAL_THRESHOLD) & (np.absolute(diffAngles) > math.pi / 2), True, False)
    found = bitstring.BitArray(binary).find(pattern)
    if len(found) == 0:
        return False
    return found[0]

"""
Demodulate and decode symbols
"""
def demodulate(symbols):
    # Demodulate symbols (to bits)
    bits = []
    # Calculate angles from previous symbols
    cur = symbols[1:] # current symbols
    prev = symbols[0:-1] # previous symbols
    prev = np.where(prev==0, prev + 1e-9, prev) # to avoid zero division
    diffAngles = np.angle(cur / prev)
    prev = False
    for a in np.absolute(diffAngles):
        if a > math.pi / 2:
            prev = not prev
        bits.append(prev)

    # Convert to binary string
    binary = bitstring.BitArray(bits)

    # Find the preamble + SFD
    pattern1 = bitstring.BitArray(hex='aa') + SFD
    pattern2 = ~pattern1
    found1 = binary.find(pattern1)
    found2 = binary.find(pattern2)
    if not found1 and not found2:
        return False
    if not found1 or found2 < found1:
        data = ~binary[found2[0]+pattern2.len:]
    else:
        data = binary[found1[0]+pattern1.len:]

    return data

"""
Call the main routine
"""
if __name__ == "__main__":
    # Parse the arguments
    args = parser.parse_args()
    main(args)

