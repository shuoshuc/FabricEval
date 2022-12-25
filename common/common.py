from common.flags import VERBOSE


def PRINTV(verbose, logstr):
    '''
    Print helper with verbosity control.
    '''
    if VERBOSE >= verbose:
        print(logstr, flush=True)
