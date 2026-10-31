from cicerone.cli import configure_logging
from cicerone.serve.app import main

if __name__ == "__main__":
    configure_logging(None, None)
    main()
