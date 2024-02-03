import ctypes
import time
import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

def run_side(trs_file, method, path):
    print("trs_file:",trs_file)
    print("path",path)
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
            
    elif method == "hpa":
        lib.hpa_.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        try:
            starttime = time.clock()
        except:
            starttime = time.perf_counter()
        lib.hpa_(trs_file.encode('utf-8'),path.encode('utf-8'))
        try:
            endtime = time.clock()
        except:
            endtime = time.perf_counter()
    elif method == "ttest":
        lib.ttest_.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        try:
            starttime = time.clock()
        except:
            starttime = time.perf_counter()
        lib.ttest_(trs_file.encode('utf-8'),path.encode('utf-8'))
        try:
            endtime = time.clock()
        except:
            endtime = time.perf_counter()
    elif method == "x2test":
        lib.x2test_.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        try:
            starttime = time.clock()
        except:
            starttime = time.perf_counter()
        lib.x2test_(trs_file.encode('utf-8'),path.encode('utf-8'))
        try:
            endtime = time.clock()
        except:
            endtime = time.perf_counter()
    elif method == "spa":
        lib.spa_.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        try:
            starttime = time.clock()
        except:
            starttime = time.perf_counter()
        lib.spa_(trs_file.encode('utf-8'),path.encode('utf-8'))
        try:
            endtime = time.clock()
        except:
            endtime = time.perf_counter()
    else:
        print(f"不支持改算法{method}")
    return endtime-starttime