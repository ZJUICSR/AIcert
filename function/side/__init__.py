import ctypes
import time
import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))
# ll = ctypes.cdll.LoadLibrary
# lib = ll('./CMake_Research3/libresreach3.so')
# lib.cpa_.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
# lib.dpa_.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
# # lib.test.restype = ctypes.c_int
# inFile = './CMake_Research3/Trs/samples/cpa/elmotrace-9/elmotracegaus_cpa_-9.trs'
# output_path = './CMake_Research3/Trs/samples/cpa/elmotrace-9/cpa_out-9.txt'
# dpa_infile = './CMake_Research3/Trs/samples/dpa/elmotrace-9/elmotracegaus_dpa_-9.trs'
# dpa_output_path = './CMake_Research3/Trs/samples/dpa/elmotrace-9/dpa_test_out-9.txt'

# for i in range(100000):
#     output_path = f'./CMake_Research3/Trs/samples/cpa/elmotrace-9/cpa_out_{i}.txt'
#     lib.cpa_(inFile.encode('utf-8'),output_path.encode('utf-8'))
#     print("次数：",i)

# print("end print")

def run_side(trs_file, method, path):
    ll = ctypes.cdll.LoadLibrary
    lib = ll(osp.join(ROOT,'CMake_Research3/libresreach3.so'))
    if method == "cpa":
        lib.cpa_.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        try:
            starttime = time.clock()
        except:
            starttime = time.perf_counter()
        lib.cpa_(trs_file.encode('utf-8'),path.encode('utf-8'))
        try:
            endtime = time.clock()
        except:
            endtime = time.perf_counter()
        
    elif method == "dpa":
        lib.dpa_.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        try:
            starttime = time.clock()
        except:
            starttime = time.perf_counter()
        lib.dpa_(trs_file.encode('utf-8'),path.encode('utf-8'))
        try:
            endtime = time.clock()
        except:
            endtime = time.perf_counter()
    return endtime-starttime