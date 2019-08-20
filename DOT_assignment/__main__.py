
import os
import atexit
from time import time, strftime, localtime
from datetime import timedelta, datetime

from DOT_assignment.main import main
from DOT_assignment.log import save_test_info_to_txt


def secondsToStr(elapsed=None):

    """ Converts seconds to strings

    """

    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))

def log(s, elapsed=None):

    """ start logging of elapsed time

    """

    line = "="*40
    print(line)
    print(secondsToStr(), '-', s)
    ss = secondsToStr() + ' - ' + s
    if elapsed:
        print("Elapsed time:", elapsed)
    print(line)
    print()
    return ss

def endlog():

    """ end log of elapsed time

    """

    end = time()
    elapsed = end-start
    log("End Program", secondsToStr(elapsed))


if __name__ == "__main__":

    start = time()
    # atexit.register(endlog) # print end time at program termination
    starttime = log("Start Program")

    # PERFORM TEST
    test_conditions = main()

    ensemble_name = test_conditions['ensemble_name']
    ensemble_directory = test_conditions['ensemble_directory']

    # PRINT TEST INFO TO TERMINAL
    print()
    line = "*"*40
    print(line)
    for condition, value in test_conditions.items():
        print(condition, ': ', value)
    print(line)
    print()

    # display starttime at the end as well as beginning
    line = "="*40
    print(line)
    print(starttime)
    print(line)
    print()

    end = time() # print end time at end of simulation
    elapsed = end-start
    elapsedstr = secondsToStr(elapsed)
    endtime = log("End Program", elapsedstr)

    save_test_info_to_txt(ensemble_name, test_conditions, ensemble_directory, starttime, endtime, elapsedstr)

