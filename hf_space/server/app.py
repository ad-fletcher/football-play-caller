"""
FastAPI application for the Football Drive Environment.

Creates an HTTP server exposing the FootballDriveEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.
"""

try:
    from openenv.core.env_server.http_server import create_app
    from ..models import GameAction, GameObservation
    from .environment import FootballDriveEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from models import GameAction, GameObservation
    from server.environment import FootballDriveEnvironment

app = create_app(
    FootballDriveEnvironment, GameAction, GameObservation,
    env_name="football_drive",
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
