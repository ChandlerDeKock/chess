# Chess (Python + FastAPI)

A basic chess game with Python logic (using `python-chess`) and a minimal web front end. The server hosts both the API and the static front end.

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the server (serves API and front end):

```bash
uvicorn backend.main:app --reload
```

3. Open the front end:

- Visit `http://localhost:8000/` in your browser.

## Features

- New game, make moves, undo.
- Legal move highlighting.
- Promotion support.
- Check / checkmate / stalemate indicators.

## API (basic)

- `GET /api/state` → current board state and game status
- `POST /api/move` → body: `{ "from": "e2", "to": "e4", "promotion": "q" }`
- `GET /api/legal-moves?from=e2` → list of targets for the square
- `POST /api/new-game` → reset to initial position
- `POST /api/undo` → undo last move

## Notes

- This app keeps a single in-memory game. For multi-user support, introduce session IDs and persist games accordingly.

