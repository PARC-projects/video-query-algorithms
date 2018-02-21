"""Brokers Queries to downstream logic based on Query.ProcessState.

This script is designed to be executed as a long running service.

TODO: Consider shifting to a daemonize approach to manage this process.
"""
import threading
import logging
import status
from datetime import datetime

LOOP_EXECUTION_TIME = 2.0 # In seconds
LOG_NAME = 'logs/query_broker_{0}.log'.format(
    datetime.now().strftime("%Y_%m_%d"))

logging.basicConfig(
    format=
    '%(asctime)s; %(levelname)s; {%(module)s}; [%(funcName)s] %(message)s',
    level=logging.DEBUG,
    handlers=[logging.FileHandler(LOG_NAME),
              logging.StreamHandler()])


def broker():
    '''Execute long pooling loop'''
    threading.Timer(LOOP_EXECUTION_TIME, broker).start()
    logging.debug('This message should go to the log file and stdOut')
    logging.info('So should this')
    logging.warning('And this, too')


if __name__ == '__main__':
    broker()
