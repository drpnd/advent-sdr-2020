import SoapySDR
import sys
import argparse
import math
import numpy as np

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--sample-rate', type=int, default=1e6)
parser.add_argument('--samples-per-symbol', type=int, default=10)
parser.add_argument('--bandwidth', type=int, default=5e6)
parser.add_argument('--rf', type=int, default=2420e6)

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

    # Setup the Rx channel 0
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, args.sample_rate)
    sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, args.bandwidth)
    sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, "LNAW")
    sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 30.0)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, args.rf)

    # Setup an Rx stream
    rxStream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [0])
    # Activate the stream
    sdr.activateStream(rxStream)

    # Check the maximum transmit unit
    mtu = sdr.getStreamMTU(rxStream)

    # Prepare a receive buffer
    rxBuffer = np.zeros(mtu, np.complex64)

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

