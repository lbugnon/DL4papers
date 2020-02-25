# ==============================================================================
# sinc(i) - http://sinc.unl.edu.ar/
# L. Bugnon, C. Yones, J. Raad, M. Gerard, M. Rubiolo, G. Merino, M. Pividori,
# L. Di Persia, D.H. Milone, and G. Stegmayer.
# lbugnon@sinc.unl.edu.ar
# ==============================================================================
import os


class Logger:
    """A simple text logger"""
    def __init__(self, dir_name="./"):
        self.out_dir = dir_name
        self.fout = dict()
        self.current_log = ""
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        global logger
        logger = self

    def start(self, file_name="log"):
        self.fout[file_name] = open(self.out_dir+file_name+".log", 'a')
        self.current_log = file_name

    def log(self, msg, log_name=None, verbose=True):
        if not log_name:
            log_name = self.current_log
        if log_name not in self.fout.keys():
            self.fout[log_name] = open(self.out_dir+log_name+".log", 'a')
        self.fout[log_name].write(msg)
        self.fout[log_name].flush()
        if verbose:
            print(msg, end='')

    def close(self, log_name=None):
        if log_name is None:
            for k in self.fout.keys():
                self.fout[k].close()
        else:
            self.fout[log_name].close()
