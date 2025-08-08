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
- Last move highlighting (from square, to square, and moved piece)
- Stronger AI (iterative deepening + TT + quiescence + improved eval)
- Human vs AI flow: human moves, server replies asynchronously; UI polls until AI completes
- AI vs AI flow: available via `/api/ai-step?force=true` (used by training)

## API (basic)

- `GET /api/state` → current board state and game status
- `POST /api/move` → body: `{ "from": "e2", "to": "e4", "promotion": "q" }`
- `GET /api/legal-moves?from=e2` → list of targets for the square
- `POST /api/new-game` → reset to initial position
- `POST /api/undo` → undo last move

### AI Engine

The AI uses iterative deepening alpha-beta with:
- Transposition table
- Quiescence search (captures at leaf nodes)
- Improved evaluation: PSTs, bishop pair, pawn structure (doubled/isolated/passed), rook open/semi-open files, king pawn shield, and mobility
- Move ordering: TT/PV move, MVV-LVA captures, killer moves, history heuristic
- Time-based search per difficulty

Evaluation weights are kept in `backend/ai_weights.json`, loaded at startup.

### Optional: Self-play Training

Background training can refine weights via self-play and saves to `backend/ai_weights.json`.

Endpoints (for training/AI-vs-AI):
- `POST /api/train-start` (params: `iterations`, `games_per_iter`, `sigma`)
- `GET /api/train-status`
- `POST /api/train-stop`
- `POST /api/ai-step?force=true` — advance the AI one move regardless of turn

Example:

```bash
curl -X POST 'http://localhost:8000/api/train-start' -H 'Content-Type: application/json' -d '{"iterations":30, "games_per_iter":6, "sigma":4.0}'
curl 'http://localhost:8000/api/train-status'
curl -X POST 'http://localhost:8000/api/train-stop'
```

## Notes

- This app keeps a single in-memory game. For multi-user support, introduce session IDs and persist games accordingly.

