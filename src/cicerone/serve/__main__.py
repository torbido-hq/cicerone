import logging

from cicerone.config.constants import DEFAULT_LOG_FORMAT
from cicerone.serve.app import main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)
    main()
