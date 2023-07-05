import ctypes
import time
ll = ctypes.cdll.LoadLibrary
lib = ll('./CMake_Research3/libresreach3.so')
# lib.cpa.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dpa_.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
# lib.test.restype = ctypes.c_int
# inFile = './CMake_Research3/Trs/samples/cpa/elmotrace-9/elmotracegaus_cpa_-9.trs'
# output_path = './CMake_Research3/Trs/samples/cpa/elmotrace-9/cpa_out-9.txt'
dpa_infile = './CMake_Research3/Trs/samples/dpa/elmotrace-9/elmotracegaus_dpa_-9.trs'
dpa_output_path = './CMake_Research3/Trs/samples/dpa/elmotrace-9/dpa_test_out-9.txt'
# method = "SPA"
# lib.cpa(inFile.encode('utf-8'),output_path.encode('utf-8'))
# start_time = time.clock()
try:
    lib.dpa_(dpa_infile.encode('utf-8'),dpa_output_path.encode('utf-8'))
except:
    pass
# end = time.clock()
# use_time = end-start_time
# print("time:",use_time)
print("end print")