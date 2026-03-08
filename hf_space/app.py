"""Entry point — re-exports the FastAPI app from server.app."""
from server.app import app, main

if __name__ == "__main__":
    main()
