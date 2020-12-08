import SoapySDR
import sys
import argparse
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

    # Setup a Rx stream
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
            samples = rxBuffer
            status = sdr.readStream(rxStream, [rxBuffer], rxBuffer.size)
            if status.ret != rxBuffer.size:
                sys.stderr.write("Failed to receive samples in readStream(): {}\n".format(status.ret))
                return False
            samples = np.concatenate([samples, rxBuffer])
            amps = np.hypot(samples.real, samples.imag)
            angles = np.arctan2(samples.imag, samples.real)
            # Plot
            ts = np.arange(samples.size)
            fig = plt.figure()
            plt.plot(ts, amps, 'g', label='r')
            plt.plot(ts, angles, 'bo-', label='theta')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude / Phase')
            plt.savefig('figure.png')
            break
        prevSamples = rxBuffer

    # Deactivate and close the stream
    sdr.deactivateStream(rxStream)
    sdr.closeStream(rxStream)


"""
Call the main routine
"""
if __name__ == "__main__":
    # Parse the arguments
    args = parser.parse_args()
    main(args)

