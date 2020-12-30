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

PREAMBLE = bitstring.BitArray(hex='aa') * 16
SFD = bitstring.BitArray(hex='2bd4')
MODULATION_BPSK = bitstring.BitArray(hex='01')
BYTE_ZERO = bitstring.BitArray(hex='00')

FRAME_TYPE_DATA = bitstring.BitArray(hex='00')

SAMPLE_RATE = 1e6
SAMPLES_PER_SYMBOL = 10

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


    # Setup Tx/Rx streams
    txStream = sdr.setupStream(SoapySDR.SOAPY_SDR_TX, SoapySDR.SOAPY_SDR_CF32, [0])
    rxStream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [0])

    # Check the maximum transmit unit
    mtu = sdr.getStreamMTU(rxStream)

    # Prepare a receive buffer
    rxBuffer = np.zeros(mtu, np.complex64)

    # Activate the streams
    sdr.activateStream(txStream)
    sdr.activateStream(rxStream)


    prevSamples = np.array([], dtype=np.complex64)
    while True:
        # Receive samples
        status = sdr.readStream(rxStream, [rxBuffer], rxBuffer.size)
        if status.ret != rxBuffer.size:
            sys.stderr.write("Failed to receive samples in readStream(): {}\n".format(status.ret))
            return False
        # Concatenate the previous samples and the current samples
        samples = np.concatenate([prevSamples, rxBuffer])
        # Calculate angles from previous symbols
        cur = samples[args.samples_per_symbol:] # current symbols
        prev = samples[0:-args.samples_per_symbol] # previous symbols
        prev = np.where(prev==0, prev + 1e-9, prev) # to avoid zero division
        x = cur / prev
        diffAngles = np.arctan2(x.imag, x.real)
        # Check if the preamble is included
        if np.sum(np.absolute(diffAngles) > math.pi / 2) >= 8 * args.samples_per_symbol:
            # Concatenate two sample buffers
            samples = np.copy(rxBuffer)
            status = sdr.readStream(rxStream, [rxBuffer], rxBuffer.size)
            if status.ret != rxBuffer.size:
                sys.stderr.write("Failed to receive samples in readStream(): {}\n".format(status.ret))
                return False
            samples = np.concatenate([samples, rxBuffer])
            if demodulate(samples, args.samples_per_symbol):
                break
        prevSamples = np.copy(rxBuffer)

    # Deactivate and close the stream
    sdr.deactivateStream(rxStream)
    sdr.closeStream(rxStream)

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
    # Build the datalink layer frame
    frame = build_datalink(dst, src, seqno, bitstring.BitArray(data))
    # Build the physical layer protocol header
    phy = build_phy(frame.size)
    # Combine the physical layer header and the data-link frame
    symbols = np.concatenate([phy, frame])

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
    hdr = MODULATION_BPSK + BYTE_ZERO + length
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
    shifted = samples[1:]
    orig = samples[0:-1]
    orig = np.where(orig==0, orig + 1e-9, orig) # to avoid zero division
    x = shifted / orig
    diffAngles = np.arctan2(x.imag, x.real)
    # Find the change points of a phase
    changePoints = np.where(np.absolute(diffAngles) > math.pi / 2)[0]
    # Find the edge index
    bc = np.bincount(changePoints % samples_per_symbol)
    return np.argmax(bc)


"""
Demodulate
"""
def demodulate(samples, samples_per_symbol):
    # Detect the edge offset
    edgeIndex = detectEdge(samples, samples_per_symbol)
    # Decode symbols from the center signal
    symbols = samples[range(edgeIndex + 4, samples.size, samples_per_symbol)]

    # PCA
    c = np.cov(np.array([symbols.real, symbols.imag]))
    e = np.linalg.eig(c)
    v = e[1][:,np.argmax(e[0])]
    base = v[0] + 1j * v[1]
    if base == 0:
        base = 1e-9

    # Demodulate the symbols
    demod = symbols / base
    binary = np.where(demod.real >= 0, True, False)

    # Detect the preamble
    psync = np.array([False, True, False, True, False, True, True, True, False, False, True, False, True, False, True, True, True, True, False, True, False, True, False, False])
    bstr = binary.tostring()
    try:
        idx0 = bstr.index(psync.tostring())
    except:
        idx0 = -1
    try:
        idx1 = bstr.index(np.logical_not(psync).tostring())
    except:
        idx1 = -1

    if idx0 < 0 and idx1 < 0:
        return False
    elif idx0 < 0:
        idx = idx1
        binary = np.logical_not(binary)
    elif idx1 < 0 or idx0 < idx1:
        idx = idx0
    else:
        idx = idx1
        binary = np.logical_not(binary)

    sys.stdout.write("Received data:")
    for i in range(16):
        if binary[idx + psync.size + i]:
            sys.stdout.write(" 1")
        else:
            sys.stdout.write(" 0")
    print("")

    return True


"""
Call the main routine
"""
if __name__ == "__main__":
    # Parse the arguments
    args = parser.parse_args()
    main(args)

