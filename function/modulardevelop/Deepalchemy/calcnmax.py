import pickle


def get_svr():
    with open('svr.pickle', 'rb') as f:
        svr = pickle.load(f)
    return svr
