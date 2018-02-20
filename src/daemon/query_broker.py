"""Brokers Queries to downstream logic based on Query.ProcessState.

This script is designed to be executed as a long running service.

TODO: Consider shifting to a daemonize approach to manage this process.
"""
import threading
import logging
from datetime import datetime

LOG_NAME = 'logs/query_broker_{0}.log'.format(
    datetime.now().strftime("%Y_%m_%d"))

logging.basicConfig(
    format='%(asctime)s; %(levelname)s; {%(module)s}; [%(funcName)s] %(message)s',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(LOG_NAME),
        logging.StreamHandler()
    ])

def broker():
    threading.Timer(2.0, broker).start()  # called every 2 seconds
    logging.debug('This message should go to the log file')
    logging.info('So should this')
    logging.warning('And this, too')


broker()
