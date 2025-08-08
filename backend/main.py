from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import chess
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import JSONResponse

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(title="Chess API + Frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory single-board state
board = chess.Board()


class MoveRequest(BaseModel):
    from_square: str
    to_square: str
    promotion: Optional[str] = None  # one of q, r, b, n


PIECE_SYMBOL_TO_NAME = {
    "p": "pawn",
    "n": "knight",
    "b": "bishop",
    "r": "rook",
    "q": "queen",
    "k": "king",
}

PROMOTION_SYMBOL_TO_TYPE = {
    "q": chess.QUEEN,
    "r": chess.ROOK,
    "b": chess.BISHOP,
    "n": chess.KNIGHT,
}


@app.get("/api/state")
def get_state() -> Dict:
    pieces: List[Dict] = []
    for square_index in chess.SQUARES:
        piece = board.piece_at(square_index)
        if piece is None:
            continue
        pieces.append(
            {
                "square": chess.square_name(square_index),
                "symbol": piece.symbol(),  # lowercase black, uppercase white
                "type": PIECE_SYMBOL_TO_NAME[piece.symbol().lower()],
                "color": "white" if piece.color else "black",
            }
        )

    return {
        "fen": board.fen(),
        "turn": "white" if board.turn else "black",
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "pieces": pieces,
        "history": [m.uci() for m in board.move_stack],
    }


@app.get("/api/legal-moves")
def get_legal_moves(from_square: str = Query(..., alias="from")) -> Dict:
    try:
        src = chess.SQUARE_NAMES.index(from_square.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid 'from' square")

    targets = []
    promotions = set()
    for move in board.legal_moves:
        if move.from_square == src:
            targets.append(chess.square_name(move.to_square))
            if move.promotion:
                promotions.add(chess.square_name(move.to_square))

    return {
        "from": from_square.lower(),
        "targets": sorted(set(targets)),
        "requiresPromotionOn": sorted(promotions),
    }


@app.post("/api/move")
def post_move(req: MoveRequest) -> Dict:
    src_name = req.from_square.lower()
    dst_name = req.to_square.lower()

    # Validate squares
    if src_name not in chess.SQUARE_NAMES or dst_name not in chess.SQUARE_NAMES:
        raise HTTPException(status_code=400, detail="Invalid square coordinates")

    src = chess.SQUARE_NAMES.index(src_name)
    dst = chess.SQUARE_NAMES.index(dst_name)

    # Gather candidate legal moves for the from->to pair
    candidates = [
        m for m in board.legal_moves if m.from_square == src and m.to_square == dst
    ]

    if not candidates:
        raise HTTPException(status_code=400, detail="Illegal move")

    chosen: Optional[chess.Move] = None

    if len(candidates) == 1 and candidates[0].promotion is None:
        chosen = candidates[0]
    else:
        # Need promotion selection
        if req.promotion is None:
            return JSONResponse(
                status_code=409,
                content={
                    "requires_promotion": True,
                    "choices": ["q", "r", "b", "n"],
                },
            )
        promo_symbol = req.promotion.lower()
        if promo_symbol not in PROMOTION_SYMBOL_TO_TYPE:
            raise HTTPException(status_code=400, detail="Invalid promotion piece")
        promo_type = PROMOTION_SYMBOL_TO_TYPE[promo_symbol]
        # Find candidate with matching promotion
        for m in candidates:
            if m.promotion == promo_type:
                chosen = m
                break
        if chosen is None:
            raise HTTPException(status_code=400, detail="Invalid promotion for move")

    # Apply move
    board.push(chosen)
    return get_state()


@app.post("/api/new-game")
def post_new_game() -> Dict:
    board.reset()
    return get_state()


@app.post("/api/undo")
def post_undo() -> Dict:
    if len(board.move_stack) == 0:
        raise HTTPException(status_code=400, detail="No moves to undo")
    board.pop()
    return get_state()


# Serve static frontend
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
