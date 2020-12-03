import SoapySDR
import sys

# Create an SDR device instance
args = dict(driver="lime")
try:
    sdr = SoapySDR.Device(args)
except:
    sys.stderr.write("Failed to create an SDR device instance.\n")
    sys.exit(1)

if not sdr:
    sys.stderr.write("Could not find any SDR devices.\n")
    sys.exit(1)

# Get the number of channels for TX/RX
n_tx_ch = sdr.getNumChannels(SoapySDR.SOAPY_SDR_TX)
n_rx_ch = sdr.getNumChannels(SoapySDR.SOAPY_SDR_TX)
print("# of channels: (TX, RX) = ({}, {})".format(n_tx_ch, n_rx_ch))

print("---")

print("Antennas:")
for i in range(n_tx_ch):
    ants = sdr.listAntennas(SoapySDR.SOAPY_SDR_TX, i)
    print("\tTX ch{}: {}".format(i, ants))
for i in range(n_rx_ch):
    ants = sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, i)
    print("\tRX ch{}: {}".format(i, ants))

print("---")

print("Gains:")
for i in range(n_tx_ch):
    g = sdr.listGains(SoapySDR.SOAPY_SDR_TX, i)
    print("\tTX ch{}:".format(i))
    for e in g:
        r = sdr.getGainRange(SoapySDR.SOAPY_SDR_TX, i, e)
        print("\t\t{}: [{}]".format(e, r))
for i in range(n_rx_ch):
    g = sdr.listGains(SoapySDR.SOAPY_SDR_RX, i)
    print("\tRX ch{}:".format(i))
    for e in g:
        r = sdr.getGainRange(SoapySDR.SOAPY_SDR_RX, i, e)
        print("\t\t{}: [{}]".format(e, r))

print("---")

print("Frequencies:")
for i in range(n_tx_ch):
    f = sdr.listFrequencies(SoapySDR.SOAPY_SDR_TX, i)
    print("\tTX ch{}:".format(i))
    for e in f:
        r = sdr.getFrequencyRange(SoapySDR.SOAPY_SDR_TX, i, e)
        print("\t\t{}: [{}]".format(e, r[0]))
for i in range(n_rx_ch):
    f = sdr.listFrequencies(SoapySDR.SOAPY_SDR_RX, i)
    print("\tRX ch{}:".format(i))
    for e in f:
        r = sdr.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, i, e)
        print("\t\t{}: [{}]".format(e, r[0]))

