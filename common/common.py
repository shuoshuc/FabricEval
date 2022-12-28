import common.flags as FLAG


def PRINTV(verbose, logstr):
    '''
    Print helper with verbosity control.
    '''
    if FLAG.VERBOSE >= verbose:
        print(logstr, flush=True)
