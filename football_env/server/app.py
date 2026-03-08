import uvicorn
from openenv.core.env_server import create_app
from .environment import FootballEnvironment
from ..models import FootballAction, FootballObservation

app = create_app(FootballEnvironment, FootballAction, FootballObservation, env_name="football_env")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
