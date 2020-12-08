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

PREAMBLE = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1]
SYNC = [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0] # Unique word
DATA = [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]

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
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_TX, 0, args.sample_rate)
    sdr.setBandwidth(SoapySDR.SOAPY_SDR_TX, 0, args.bandwidth)
    sdr.setAntenna(SoapySDR.SOAPY_SDR_TX, 0, "BAND1")
    sdr.setGain(SoapySDR.SOAPY_SDR_TX, 0, 50.0)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_TX, 0, args.rf)

    # Setup a Tx stream
    txStream = sdr.setupStream(SoapySDR.SOAPY_SDR_TX, SoapySDR.SOAPY_SDR_CF32, [0])
    # Activate the stream
    sdr.activateStream(txStream)

    # Data to send
    data = []
    for d in PREAMBLE + SYNC + DATA:
        data += [d] * args.samples_per_symbol

    # Check the maximum transmit unit
    mtu = sdr.getStreamMTU(txStream)
    if mtu < len(data):
        sys.stderr.write("MTU is too small {} < {}\n".format(mtu, len(data)))
        return False

    # Build preamble, sync code (unique word) and data
    samples = np.exp( 1j * math.pi * np.array(data, np.complex64) ).astype(np.complex64)

    # Build 0...0 (1 + 0j)s
    base = np.ones(mtu, np.complex64)

    # Transmit the base signal (to warm up)
    for i in range(10000):
        status = sdr.writeStream(txStream, [base], base.size, timeoutUs=1000000)
        if status.ret != base.size:
            sys.stderr.write("Failed to transmit all samples in writeStream(): {}\n".format(status.ret))
            return False

    # Transmit the samples
    print("Sending data...")
    status = sdr.writeStream(txStream, [samples], samples.size, SoapySDR.SOAPY_SDR_END_BURST, timeoutUs=1000000)
    if status.ret != samples.size:
        sys.stderr.write("Failed to transmit all samples in writeStream(): {}\n".format(status.ret))
        return False

    # Transmit the base signal
    for i in range(10000):
        status = sdr.writeStream(txStream, [base], base.size, timeoutUs=1000000)
        if status.ret != base.size:
            sys.stderr.write("Failed to transmit all samples in writeStream(): {}\n".format(status.ret))
            return False

    # Deactivate and close the stream
    sdr.deactivateStream(txStream)
    sdr.closeStream(txStream)


"""
Call the main routine
"""
if __name__ == "__main__":
    # Parse the arguments
    args = parser.parse_args()
    main(args)

