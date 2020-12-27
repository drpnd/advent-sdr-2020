import SoapySDR
import sys
import argparse
import math
import numpy as np
import bitstring
import binascii
import struct

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--bandwidth', type=int, default=5e6)
parser.add_argument('--rf', type=int, default=2420e6)
parser.add_argument('--tx-gain', type=float, default=50.0)
parser.add_argument('--tx-antenna', type=str, default='BAND1')
parser.add_argument('--rx-gain', type=float, default=30.0)
parser.add_argument('--rx-antenna', type=str, default='LNAW')

PREAMBLE = bitstring.BitArray(hex='aa') * 16
SYNC = bitstring.BitArray(hex='2bd4')
MODULATION_BPSK = bitstring.BitArray(hex='01')
ZERO_BYTE = bitstring.BitArray(hex='00')

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
    sdr.setGain(SoapySDR.SOAPY_SDR_TX, 0, args.tx_gain)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, args.rf)

    # Setup the Rx channel 0
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, SAMPLE_RATE)
    sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, args.bandwidth)
    sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, args.rx_antenna)
    sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, args.rx_gain)
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
    #f = crcmod.mkCrcFun(0x11021, rev=False, initCrc=0x1d0f, xorOut=0x0000)
    f = crcmod.predefined.mkPredefinedCrcFun('crc-aug-ccitt')
    return f(bstr)
def crc16_checksum(bstr):
    return crc16(bstr).to_bytes(2, 'big')
    #return struct.pack('>H', crc16(bstr))
def crc16_check(bstr):
    if crc16(bstr) == 0:
        return True
    else:
        return False

"""
CRC-32
"""
def crc32(bstr):
    #f = crcmod.mkCrcFun(0x104c11db7, rev=True, initCrc=0x00000000, xorOut=0xffffffff)
    f = crcmod.predefined.mkPredefinedCrcFun('crc32')
    return f(bstr)
def crc32_str(bstr):
    return crc32(bstr).to_bytes(4, 'little')
    #return struct.pack('<L', crc32(bstr))
def crc32_check(bstr):
    if crc32(bstr) == 0x2144df1c:
        return True
    else:
        return False

"""
Transmit
"""
def tx_packet(sdr, data):
    # Physical layer
    PREAMBLE + SYNC + MODULATION_BPSK + RESERVED + ZERO_BYTE
    # CRC-16
    
    return True


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
Modulate
"""
def modulate_bpsk(data, samples_per_symbol):
    samples = []
    for b in data:
        samples += [b] * samples_per_symbol
    return np.exp( 1j * math.pi * np.array(samples, np.complex64) ).astype(np.complex64)

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

