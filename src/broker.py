"""Brokers Queries to downstream logic based on Query.ProcessState.

This script is designed to be executed as a long running service.

TODO: Consider shifting to a daemonize approach to manage this process.
"""
import threading
import logging
import os
import time
from api.status import QueryStatus
from datetime import datetime
from models import compute_matches

LOOP_EXECUTION_TIME = 10.0  # In seconds
LOG_NAME = 'logs/query_broker_{0}.log'.format(
    datetime.now().strftime("%Y_%m_%d"))
FORMAT = '%(asctime)s; %(levelname)s; {%(module)s}; [%(funcName)s] %(message)s'

logging.basicConfig(
    format=FORMAT,
    level=logging.DEBUG,
    handlers=[logging.FileHandler(LOG_NAME),
              logging.StreamHandler()])


def main():
    '''Execute long pooling loop'''
    queryStatus = QueryStatus()

    try:
        result = queryStatus.getStatus()
        if result["new"]:
            compute_matches.new_matches(result["new"])
        if result["revise"]:
            compute_matches.revised_matches(result["revise"], [])
    except Exception as e:
        logging.error(e)
    finally:
        # TODO: Discuss options (thoughts)
        # We can do a number of things in this process.
        # - Thread so we have non-blocking operations. Some reasoning...
        #   - We want to run compute processes in parallel.
        #   - We want to run a bit of logic after we send data down-stream but
        #       don't want to wait for down-stream processing to return.
        # - Block operations until down-stream is done.
        #   - Simple and traceable.
        #   - Would we want to throttle the synchronous process?
        #       - If not, there would be no value in setting a sleep to throttle.

        # TODO: Is BROKER_IS_THREADED a good name?
        if os.environ.get('BROKER_IS_THREADED') == 'True':
            # create a new thread
            threading.Timer(LOOP_EXECUTION_TIME, main).start()
        else:
            # time.sleep(LOOP_EXECUTION_TIME)
            main()


if __name__ == '__main__':
    main()
