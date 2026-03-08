import uvicorn
from openenv.core.env_server import create_app
from environment import FootballDriveEnvironment
from models import GameAction, GameObservation

app = create_app(FootballDriveEnvironment, GameAction, GameObservation, env_name="football_drive")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
