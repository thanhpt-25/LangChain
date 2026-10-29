"""Print the URL of the most recent root run in the finassist project."""
import os

from langsmith import Client

from finassist.config import get_settings


def main() -> None:
    get_settings()  # loads .env
    client = Client()
    project = os.getenv("LANGSMITH_PROJECT", "finassist-w1")
    run = next(iter(client.list_runs(project_name=project, is_root=True, limit=1)), None)
    if run is None:
        raise SystemExit(f"no runs in project {project}")
    print(getattr(run, "url", None) or client.read_run(run.id).url)


if __name__ == "__main__":
    main()
