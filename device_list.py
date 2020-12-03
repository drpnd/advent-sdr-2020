import SoapySDR

# Enumerate all SDR devices
print("Obtain the recognized SDR devices...")
results = SoapySDR.Device.enumerate()
print("Found {} devices:".format(len(results)))
for result in results: print(result)

