#!/usr/bin/env python3
import threading
import subprocess
import time
import datetime
import sys

#print("The Python version is {}".format( sys.version_info))

def popenAndCall(my_list, popenArgs):
    """
    Runs the given args in a subprocess.Popen, and then calls the function
    onExit when the subprocess completes.
    onExit is a callable object, and popenArgs is a list/tuple of args that
    would give to subprocess.Popen.
    """
    def runInThread(my_list, popenArgs):
        proc = subprocess.Popen(*popenArgs, shell=True)
        proc.wait()
        my_list[0](my_list[1])
        return
    thread = threading.Thread(target=runInThread, args=(my_list, popenArgs))
    thread.start()
    # returns immediately after the thread starts
    return thread

def exit_func(count):
    print("Count".format(count))
    now = datetime.datetime.now()
    if now < end_time:
        print("Countinue: Current time {} is less than {}".format(now, end_time))
        popenAndCall([exit_func, count], [sys.executable + " run_locally.py"])
    else:
        print("End: Current time {} is greater than {}".format(now, end_time))
        print("All Done")


start_time = datetime.datetime.now()

print("Start Time: {}".format(start_time))

end_time = start_time + datetime.timedelta(hours=6)
print("Stop running at time: {}".format(end_time))

#subprocess.call("alias python3='/usr/bin/python3.6", shell=True)
popenAndCall([exit_func, 0], [sys.executable + " run_locally.py"])
popenAndCall([exit_func, 0], [sys.executable + " run_locally.py"])
popenAndCall([exit_func, 0], [sys.executable + " run_locally.py"])
popenAndCall([exit_func, 0], [sys.executable + " run_locally.py"])
popenAndCall([exit_func, 0], [sys.executable + " run_locally.py"])
popenAndCall([exit_func, 0], [sys.executable + " run_locally.py"])
# popenAndCall([exit_func, 0], [sys.executable + " run_locally.py"])
# popenAndCall([exit_func, 0], [sys.executable + " run_locally.py"])



#popenAndCall([exit_func, 0], [sys.executable + " run_locally.py"])
#popenAndCall([exit_func, 0], ["python3 run_locally.py"])
# test2 = popenAndCall(exit_func, ["python app.py"])
#
# print(test)
# print(test2)
