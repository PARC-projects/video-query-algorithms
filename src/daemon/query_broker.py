"""Brokers Queries to downstream logic based on Query.ProcessState.

This script is designed to be executed as a long running service.

"""
import threading
import logging
from datetime import datetime


logging.basicConfig(filename='logs/query_broker_{0}.log'.format(datetime.now().strftime("%Y_%m_%d")),level=logging.DEBUG)

def hello_world():
    threading.Timer(2.0, hello_world).start() # called every minute
    logging.debug('This message should go to the log file')
    logging.info('So should this')
    logging.warning('And this, too')

hello_world()
