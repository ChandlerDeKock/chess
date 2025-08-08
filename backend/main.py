from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import random

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

# In-memory single-board state and config
board = chess.Board()
human_color: str = "white"  # 'white' or 'black'
ai_difficulty: str = "medium"  # 'easy', 'medium', 'hard'


class MoveRequest(BaseModel):
    from_square: str
    to_square: str
    promotion: Optional[str] = None  # one of q, r, b, n


class ConfigRequest(BaseModel):
    human_color: Optional[str] = None  # 'white' or 'black'
    difficulty: Optional[str] = None   # 'easy', 'medium', 'hard'


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

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-square tables from white's perspective (coarse, illustrative)
# Source inspiration: chess programming wiki PSTs
PAWN_PST = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
KNIGHT_PST = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]
BISHOP_PST = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]
ROOK_PST = [
     0,  0,  5, 10, 10,  5,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
QUEEN_PST = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]
KING_PST_MID = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]

PST_BY_PIECE = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_PST_MID,
}


def pst_value(piece: chess.Piece, square_index: int) -> int:
    table = PST_BY_PIECE[piece.piece_type]
    # White perspective index; mirror for black by flipping rank
    if piece.color == chess.WHITE:
        return table[square_index]
    # Mirror across horizontal axis for black
    file = chess.square_file(square_index)
    rank = chess.square_rank(square_index)
    mirrored = chess.square(file, 7 - rank)
    return -table[mirrored]


def evaluate_board(b: chess.Board) -> int:
    # Positive: good for White; negative: good for Black
    score = 0
    # Material + piece-square
    for sq in chess.SQUARES:
        p = b.piece_at(sq)
        if p is None:
            continue
        material = PIECE_VALUES[p.piece_type]
        score += material if p.color == chess.WHITE else -material
        score += pst_value(p, sq)
    # Small mobility term for side to move
    mobility = b.legal_moves.count()
    score += 2 * mobility if b.turn == chess.WHITE else -2 * mobility
    return score


def order_moves(b: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
    def move_score(m: chess.Move) -> int:
        score = 0
        # Prioritize captures and promotions
        if b.is_capture(m):
            score += 1000
        if m.promotion:
            score += 900
        # Simple center preference
        to_sq = m.to_square
        file = chess.square_file(to_sq)
        rank = chess.square_rank(to_sq)
        if 2 <= file <= 5 and 2 <= rank <= 5:
            score += 5
        return score
    return sorted(list(moves), key=move_score, reverse=True)


def negamax(b: chess.Board, depth: int, alpha: int, beta: int, ai_is_white: bool) -> int:
    if depth == 0 or b.is_game_over():
        base = evaluate_board(b)
        return base if ai_is_white else -base

    max_eval = -10_000_000
    for move in order_moves(b, list(b.legal_moves)):
        b.push(move)
        eval_child = -negamax(b, depth - 1, -beta, -alpha, ai_is_white)
        b.pop()
        if eval_child > max_eval:
            max_eval = eval_child
        if max_eval > alpha:
            alpha = max_eval
        if alpha >= beta:
            break
    return max_eval


def select_ai_move(b: chess.Board, ai_color_white: bool) -> Optional[chess.Move]:
    global ai_difficulty
    legal = list(b.legal_moves)
    if not legal:
        return None

    # Difficulty configuration
    if ai_difficulty == "easy":
        # 20% random blunder, otherwise shallow capture-preferring choice
        if random.random() < 0.2:
            return random.choice(legal)
        best, best_val = None, -10_000_000
        for m in legal:
            b.push(m)
            val = evaluate_board(b)
            b.pop()
            val = val if ai_color_white else -val
            if b.is_capture(m):
                val += 15
            if val > best_val:
                best_val, best = val, m
        # Add small noise to avoid determinism
        if random.random() < 0.25:
            return random.choice(order_moves(b, legal)[:5])
        return best

    # Medium and hard: alpha-beta search
    depth = 2 if ai_difficulty == "medium" else 3
    best_move = None
    best_eval = -10_000_000
    alpha, beta = -10_000_000, 10_000_000

    for m in order_moves(b, legal):
        b.push(m)
        score = -negamax(b, depth - 1, -beta, -alpha, ai_color_white)
        b.pop()
        if score > best_eval:
            best_eval = score
            best_move = m
        if score > alpha:
            alpha = score

    return best_move


def is_human_turn() -> bool:
    return (board.turn and human_color == "white") or ((not board.turn) and human_color == "black")


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
        "human_color": human_color,
        "ai_difficulty": ai_difficulty,
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


@app.get("/api/config")
def get_config() -> Dict:
    return {"human_color": human_color, "difficulty": ai_difficulty}


@app.post("/api/config")
def set_config(req: ConfigRequest) -> Dict:
    global human_color, ai_difficulty
    if req.human_color is not None:
        if req.human_color not in ("white", "black"):
            raise HTTPException(status_code=400, detail="human_color must be 'white' or 'black'")
        human_color = req.human_color
    if req.difficulty is not None:
        if req.difficulty not in ("easy", "medium", "hard"):
            raise HTTPException(status_code=400, detail="difficulty must be 'easy', 'medium', or 'hard'")
        ai_difficulty = req.difficulty
    return {"human_color": human_color, "difficulty": ai_difficulty}


@app.post("/api/move")
def post_move(req: MoveRequest) -> Dict:
    if not is_human_turn():
        raise HTTPException(status_code=400, detail="Not your turn")

    src_name = req.from_square.lower()
    dst_name = req.to_square.lower()

    # Validate squares
    if src_name not in chess.SQUARE_NAMES or dst_name not in chess.SQUARE_NAMES:
        raise HTTPException(status_code=400, detail="Invalid square coordinates")

    src = chess.SQUARE_NAMES.index(src_name)
    dst = chess.SQUARE_NAMES.index(dst_name)

    # Ensure moving a human piece
    piece = board.piece_at(src)
    if piece is None:
        raise HTTPException(status_code=400, detail="No piece on source square")
    if (piece.color and human_color != "white") or ((not piece.color) and human_color != "black"):
        raise HTTPException(status_code=400, detail="Cannot move AI's piece")

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

    # Apply human move
    board.push(chosen)

    # If game not over and now it's AI's turn, make AI move
    if not board.is_game_over() and not is_human_turn():
        ai_move()

    return get_state()


def ai_move() -> Optional[str]:
    ai_color_white = board.turn  # AI moves on its turn when called
    move = select_ai_move(board, ai_color_white)
    if move is None:
        return None
    board.push(move)
    return move.uci()


@app.post("/api/new-game")
def post_new_game() -> Dict:
    board.reset()
    # If AI is white and should move first
    if not is_human_turn() and not board.is_game_over():
        ai_move()
    return get_state()


@app.post("/api/undo")
def post_undo() -> Dict:
    if len(board.move_stack) == 0:
        raise HTTPException(status_code=400, detail="No moves to undo")
    # Undo one ply; if undo results in AI to move (meaning last was AI), undo again to revert full turn
    board.pop()
    if not is_human_turn() and len(board.move_stack) > 0:
        board.pop()
    return get_state()


# Serve static frontend
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
