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
import time
import threading
import json
from dataclasses import dataclass

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

# AI weight storage (tunable)
WEIGHTS_PATH = ROOT_DIR / "backend" / "ai_weights.json"

DEFAULT_WEIGHTS: Dict[str, float] = {
    # Material multipliers (kept for potential tuning)
    "material_pawn": 100.0,
    "material_knight": 320.0,
    "material_bishop": 330.0,
    "material_rook": 500.0,
    "material_queen": 900.0,
    # Piece-square influence multiplier
    "pst_scalar": 1.0,
    # Mobility bonus per legal move for side to move
    "mobility": 2.0,
    # Bishop pair bonus
    "bishop_pair": 35.0,
    # Pawn structure
    "doubled_pawn": -18.0,
    "isolated_pawn": -12.0,
    "passed_pawn": 25.0,
    # Rooks
    "rook_open_file": 20.0,
    "rook_semi_open_file": 10.0,
    # King safety (very rough)
    "king_pawn_shield": 8.0,
}

AI_WEIGHTS: Dict[str, float] = {}


def load_ai_weights() -> None:
    global AI_WEIGHTS
    if WEIGHTS_PATH.exists():
        try:
            AI_WEIGHTS = json.loads(WEIGHTS_PATH.read_text())
        except Exception:
            AI_WEIGHTS = DEFAULT_WEIGHTS.copy()
    else:
        AI_WEIGHTS = DEFAULT_WEIGHTS.copy()
        try:
            WEIGHTS_PATH.write_text(json.dumps(AI_WEIGHTS, indent=2))
        except Exception:
            pass


def save_ai_weights() -> None:
    try:
        WEIGHTS_PATH.write_text(json.dumps(AI_WEIGHTS, indent=2))
    except Exception:
        pass


class MoveRequest(BaseModel):
    from_square: str
    to_square: str
    promotion: Optional[str] = None  # one of q, r, b, n


class ConfigRequest(BaseModel):
    human_color: Optional[str] = None  # 'white' or 'black'
    difficulty: Optional[str] = None   # 'easy', 'medium', 'hard'


class TrainStartRequest(BaseModel):
    iterations: Optional[int] = 20
    games_per_iter: Optional[int] = 4
    sigma: Optional[float] = 5.0


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
    chess.PAWN: int(DEFAULT_WEIGHTS["material_pawn"]),
    chess.KNIGHT: int(DEFAULT_WEIGHTS["material_knight"]),
    chess.BISHOP: int(DEFAULT_WEIGHTS["material_bishop"]),
    chess.ROOK: int(DEFAULT_WEIGHTS["material_rook"]),
    chess.QUEEN: int(DEFAULT_WEIGHTS["material_queen"]),
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


def count_bishops(b: chess.Board, color: bool) -> int:
    return len(b.pieces(chess.BISHOP, color))


def file_has_pawn(b: chess.Board, color: bool, file_idx: int) -> bool:
    for rank in range(8):
        sq = chess.square(file_idx, rank)
        p = b.piece_at(sq)
        if p and p.piece_type == chess.PAWN and p.color == color:
            return True
    return False


def is_doubled_pawn(b: chess.Board, color: bool, file_idx: int) -> bool:
    count = 0
    for rank in range(8):
        p = b.piece_at(chess.square(file_idx, rank))
        if p and p.piece_type == chess.PAWN and p.color == color:
            count += 1
            if count > 1:
                return True
    return False


def is_isolated_pawn(b: chess.Board, color: bool, file_idx: int) -> bool:
    left_file = max(0, file_idx - 1)
    right_file = min(7, file_idx + 1)
    # Check adjacent files for same color pawns
    for f in (left_file, right_file):
        if f == file_idx:
            continue
        if file_has_pawn(b, color, f):
            return False
    return True


def is_passed_pawn(b: chess.Board, color: bool, file_idx: int, rank_idx: int) -> bool:
    # Passed pawn: no enemy pawns on same or adjacent files ahead of it
    enemy = not color
    files_to_check = [file_idx]
    if file_idx - 1 >= 0:
        files_to_check.append(file_idx - 1)
    if file_idx + 1 <= 7:
        files_to_check.append(file_idx + 1)
    if color == chess.WHITE:
        ranks = range(rank_idx + 1, 8)
    else:
        ranks = range(0, rank_idx)
    for f in files_to_check:
        for r in ranks:
            p = b.piece_at(chess.square(f, r))
            if p and p.piece_type == chess.PAWN and p.color == enemy:
                return False
    return True


def rook_file_status(b: chess.Board, color: bool, file_idx: int) -> Tuple[bool, bool]:
    # Returns (open_file, semi_open_file)
    has_friendly_pawn = False
    has_enemy_pawn = False
    for rank in range(8):
        sq = chess.square(file_idx, rank)
        p = b.piece_at(sq)
        if p and p.piece_type == chess.PAWN:
            if p.color == color:
                has_friendly_pawn = True
            else:
                has_enemy_pawn = True
    open_file = not has_friendly_pawn and not has_enemy_pawn
    semi_open = (not has_friendly_pawn) and has_enemy_pawn
    return open_file, semi_open


def king_pawn_shield(b: chess.Board, color: bool) -> int:
    # Simple measure: count friendly pawns in the three files centered on king file, on home ranks in front of king
    king_sq = b.king(color)
    if king_sq is None:
        return 0
    kfile = chess.square_file(king_sq)
    files = [kfile]
    if kfile - 1 >= 0:
        files.append(kfile - 1)
    if kfile + 1 <= 7:
        files.append(kfile + 1)
    count = 0
    if color == chess.WHITE:
        ranks = [1, 2]
    else:
        ranks = [6, 5]
    for f in files:
        for r in ranks:
            p = b.piece_at(chess.square(f, r))
            if p and p.piece_type == chess.PAWN and p.color == color:
                count += 1
    return count


def evaluate_board_raw(b: chess.Board) -> int:
    # Positive: good for White; negative: good for Black
    w = AI_WEIGHTS
    pst_scalar = w.get("pst_scalar", 1.0)
    score = 0.0

    white_minor_count = 0
    black_minor_count = 0

    # Material and PST
    for sq in chess.SQUARES:
        p = b.piece_at(sq)
        if p is None:
            continue
        base = 0.0
        if p.piece_type == chess.PAWN:
            base = w.get("material_pawn", 100.0)
        elif p.piece_type == chess.KNIGHT:
            base = w.get("material_knight", 320.0)
        elif p.piece_type == chess.BISHOP:
            base = w.get("material_bishop", 330.0)
        elif p.piece_type == chess.ROOK:
            base = w.get("material_rook", 500.0)
        elif p.piece_type == chess.QUEEN:
            base = w.get("material_queen", 900.0)
        # King material stays 0

        pst = pst_scalar * pst_value(p, sq)
        if p.color == chess.WHITE:
            score += base + pst
        else:
            score -= base + pst

        if p.piece_type in (chess.BISHOP, chess.KNIGHT):
            if p.color == chess.WHITE:
                white_minor_count += 1
            else:
                black_minor_count += 1

    # Bishop pair
    if count_bishops(b, chess.WHITE) >= 2:
        score += w.get("bishop_pair", 35.0)
    if count_bishops(b, chess.BLACK) >= 2:
        score -= w.get("bishop_pair", 35.0)

    # Pawn structure
    for file_idx in range(8):
        # Doubled and isolated
        if is_doubled_pawn(b, chess.WHITE, file_idx):
            score += w.get("doubled_pawn", -18.0)
        if is_doubled_pawn(b, chess.BLACK, file_idx):
            score -= w.get("doubled_pawn", -18.0)
        if file_has_pawn(b, chess.WHITE, file_idx) and is_isolated_pawn(b, chess.WHITE, file_idx):
            score += w.get("isolated_pawn", -12.0)
        if file_has_pawn(b, chess.BLACK, file_idx) and is_isolated_pawn(b, chess.BLACK, file_idx):
            score -= w.get("isolated_pawn", -12.0)

        # Rook files
        open_file_w, semi_w = rook_file_status(b, chess.WHITE, file_idx)
        open_file_b, semi_b = rook_file_status(b, chess.BLACK, file_idx)
        if open_file_w:
            # for each white rook on that file
            for r_sq in b.pieces(chess.ROOK, chess.WHITE):
                if chess.square_file(r_sq) == file_idx:
                    score += w.get("rook_open_file", 20.0)
        if semi_w:
            for r_sq in b.pieces(chess.ROOK, chess.WHITE):
                if chess.square_file(r_sq) == file_idx:
                    score += w.get("rook_semi_open_file", 10.0)
        if open_file_b:
            for r_sq in b.pieces(chess.ROOK, chess.BLACK):
                if chess.square_file(r_sq) == file_idx:
                    score -= w.get("rook_open_file", 20.0)
        if semi_b:
            for r_sq in b.pieces(chess.ROOK, chess.BLACK):
                if chess.square_file(r_sq) == file_idx:
                    score -= w.get("rook_semi_open_file", 10.0)

        # Passed pawns
        for rank in range(8):
            p = b.piece_at(chess.square(file_idx, rank))
            if not p or p.piece_type != chess.PAWN:
                continue
            if p.color == chess.WHITE and is_passed_pawn(b, chess.WHITE, file_idx, rank):
                score += w.get("passed_pawn", 25.0)
            if p.color == chess.BLACK and is_passed_pawn(b, chess.BLACK, file_idx, rank):
                score -= w.get("passed_pawn", 25.0)

    # King safety pawn shield
    score += w.get("king_pawn_shield", 8.0) * king_pawn_shield(b, chess.WHITE)
    score -= w.get("king_pawn_shield", 8.0) * king_pawn_shield(b, chess.BLACK)

    # Mobility for side to move
    mobility = b.legal_moves.count()
    if b.turn == chess.WHITE:
        score += w.get("mobility", 2.0) * mobility
    else:
        score -= w.get("mobility", 2.0) * mobility

    return int(score)


def evaluate_for_side_to_move(b: chess.Board) -> int:
    base = evaluate_board_raw(b)
    return base if b.turn == chess.WHITE else -base


MVV_LVA = {
    (chess.PAWN, chess.PAWN): 105,
    (chess.PAWN, chess.KNIGHT): 205,
    (chess.PAWN, chess.BISHOP): 205,
    (chess.PAWN, chess.ROOK): 305,
    (chess.PAWN, chess.QUEEN): 405,
    (chess.KNIGHT, chess.PAWN): 104,
    (chess.KNIGHT, chess.KNIGHT): 204,
    (chess.KNIGHT, chess.BISHOP): 204,
    (chess.KNIGHT, chess.ROOK): 304,
    (chess.KNIGHT, chess.QUEEN): 404,
    (chess.BISHOP, chess.PAWN): 103,
    (chess.BISHOP, chess.KNIGHT): 203,
    (chess.BISHOP, chess.BISHOP): 203,
    (chess.BISHOP, chess.ROOK): 303,
    (chess.BISHOP, chess.QUEEN): 403,
    (chess.ROOK, chess.PAWN): 102,
    (chess.ROOK, chess.KNIGHT): 202,
    (chess.ROOK, chess.BISHOP): 202,
    (chess.ROOK, chess.ROOK): 302,
    (chess.ROOK, chess.QUEEN): 402,
    (chess.QUEEN, chess.PAWN): 101,
    (chess.QUEEN, chess.KNIGHT): 201,
    (chess.QUEEN, chess.BISHOP): 201,
    (chess.QUEEN, chess.ROOK): 301,
    (chess.QUEEN, chess.QUEEN): 401,
}


def move_order_score(b: chess.Board, m: chess.Move, tt_move: Optional[chess.Move], killers: List[List[Optional[str]]], history: Dict[Tuple[int, int], int], ply: int) -> int:
    if tt_move is not None and m == tt_move:
        return 10_000_000
    score = 0
    # Captures by MVV-LVA
    if b.is_capture(m):
        victim = b.piece_type_at(m.to_square)
        attacker = b.piece_type_at(m.from_square)
        if victim is not None and attacker is not None:
            score += 100_000 + MVV_LVA.get((attacker, victim), 100)
        else:
            score += 100_000
    # Promotions
    if m.promotion:
        score += 90_000 + (10 * m.promotion)
    # Killer moves (non-captures)
    if not b.is_capture(m) and 0 <= ply < len(killers):
        k1, k2 = killers[ply]
        u = m.uci()
        if u == k1:
            score += 50_000
        elif u == k2:
            score += 40_000
    # History heuristic
    key = (m.from_square, m.to_square)
    score += history.get(key, 0)
    return score


@dataclass
class TTEntry:
    depth: int
    value: int
    flag: int  # 0 exact, 1 lower, 2 upper
    best_move: Optional[str]


TT: Dict[int, TTEntry] = {}
KILLERS: List[List[Optional[str]]] = [[None, None] for _ in range(128)]
HISTORY: Dict[Tuple[int, int], int] = {}


def tt_key(b: chess.Board) -> int:
    # Try multiple ways to get a Zobrist-like key; fall back to FEN hash
    try:
        return int(b.transposition_key())  # type: ignore[attr-defined]
    except Exception:
        try:
            return int(b.zobrist_hash())  # type: ignore[attr-defined]
        except Exception:
            return hash(b.board_fen() + (" w" if b.turn else " b"))


class TimeAbort(Exception):
    pass


def quiescence(b: chess.Board, alpha: int, beta: int, nodes: List[int], time_deadline: float) -> int:
    if time.perf_counter() > time_deadline:
        raise TimeAbort()
    stand_pat = evaluate_for_side_to_move(b)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    # Generate captures only
    for m in b.legal_moves:
        if not b.is_capture(m):
            continue
        nodes[0] += 1
        b.push(m)
        score = -quiescence(b, -beta, -alpha, nodes, time_deadline)
        b.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def alphabeta(b: chess.Board, depth: int, alpha: int, beta: int, ply: int, nodes: List[int], time_deadline: float) -> int:
    if time.perf_counter() > time_deadline:
        raise TimeAbort()

    key = tt_key(b)
    tt_entry = TT.get(key)
    tt_move_obj: Optional[chess.Move] = None
    if tt_entry and tt_entry.depth >= depth:
        if tt_entry.flag == 0:
            return tt_entry.value
        elif tt_entry.flag == 1 and tt_entry.value > alpha:
            alpha = tt_entry.value
        elif tt_entry.flag == 2 and tt_entry.value < beta:
            beta = tt_entry.value
        if alpha >= beta:
            return tt_entry.value
    if tt_entry and tt_entry.best_move:
        try:
            tt_move_obj = chess.Move.from_uci(tt_entry.best_move)
        except Exception:
            tt_move_obj = None

    if depth == 0:
        return quiescence(b, alpha, beta, nodes, time_deadline)

    legal_moves = list(b.legal_moves)
    if not legal_moves:
        # Checkmate or stalemate
        if b.is_check():
            return -9_999_999 + ply
        return 0

    # Order moves
    ordered = sorted(
        legal_moves,
        key=lambda m: move_order_score(b, m, tt_move_obj, KILLERS, HISTORY, ply),
        reverse=True,
    )

    best_score = -10_000_000
    best_move_str: Optional[str] = None
    original_alpha = alpha

    for m in ordered:
        nodes[0] += 1
        b.push(m)
        score = -alphabeta(b, depth - 1, -beta, -alpha, ply + 1, nodes, time_deadline)
        b.pop()

        if score > best_score:
            best_score = score
            best_move_str = m.uci()
        if score > alpha:
            alpha = score

        if alpha >= beta:
            # Beta cutoff: update killers/history for non-captures
            if not b.is_capture(m):
                if KILLERS[ply][0] != m.uci():
                    KILLERS[ply][1] = KILLERS[ply][0]
                    KILLERS[ply][0] = m.uci()
                key_hist = (m.from_square, m.to_square)
                HISTORY[key_hist] = HISTORY.get(key_hist, 0) + depth * depth
            break

    # Store in TT
    flag = 0
    if best_score <= original_alpha:
        flag = 2  # upper bound
    elif best_score >= beta:
        flag = 1  # lower bound
    TT[key] = TTEntry(depth=depth, value=best_score, flag=flag, best_move=best_move_str)
    return best_score


def search_best_move(b: chess.Board, difficulty: str) -> Optional[chess.Move]:
    # Time controls per difficulty
    if difficulty == "easy":
        max_time = 0.05
        max_depth = 2
    elif difficulty == "medium":
        max_time = 0.3
        max_depth = 4
    else:
        max_time = 1.2
        max_depth = 6

    global TT, KILLERS, HISTORY
    TT = {}
    KILLERS = [[None, None] for _ in range(128)]
    HISTORY = {}

    start = time.perf_counter()
    deadline = start + max_time
    best_move: Optional[chess.Move] = None
    best_score = -10_000_000
    nodes = [0]
    last_completed_depth = 0

    legal = list(b.legal_moves)
    if not legal:
        return None

    # Iterative deepening
    for depth in range(1, max_depth + 1):
        try:
            local_best: Optional[chess.Move] = None
            local_best_score = -10_000_000
            alpha, beta = -10_000_000, 10_000_000
            # Use previous pv move as first move ordering hint if any
            ordered = legal
            if best_move is not None:
                # Move previous best to front
                try:
                    ordered = [best_move] + [m for m in legal if m != best_move]
                except Exception:
                    ordered = legal
            for m in ordered:
                nodes[0] += 1
                b.push(m)
                score = -alphabeta(b, depth - 1, -beta, -alpha, 1, nodes, deadline)
                b.pop()
                if score > local_best_score:
                    local_best_score = score
                    local_best = m
                if score > alpha:
                    alpha = score
            if local_best is not None:
                best_move = local_best
                best_score = local_best_score
                last_completed_depth = depth
            if time.perf_counter() > deadline:
                break
        except TimeAbort:
            break

    # Fallback: if no search completed, pick a legal move (random or capture pref)
    if best_move is None:
        caps = [m for m in legal if b.is_capture(m)]
        return random.choice(caps or legal)
    return best_move


# (Old negamax removed; replaced by alphabeta/quiescence with TT)


def select_ai_move(b: chess.Board, ai_color_white: bool) -> Optional[chess.Move]:
    global ai_difficulty
    # On easy, sometimes play randomly to be weaker
    if ai_difficulty == "easy" and random.random() < 0.15:
        legal = list(b.legal_moves)
        if not legal:
            return None
        return random.choice(legal)
    return search_best_move(b, ai_difficulty)


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
        "ai_thinking": ai_thinking_flag.is_set(),
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

    # If game not over and now it's AI's turn, make AI move synchronously (original flow)
    if not board.is_game_over() and not is_human_turn():
        ai_move()

    return get_state()


def ai_move() -> Optional[str]:
    # Safety: only move when it's AI's turn
    if is_human_turn() or board.is_game_over():
        return None
    ai_color_white = board.turn  # AI moves on its turn when called
    move = select_ai_move(board, ai_color_white)
    if move is None:
        return None
    board.push(move)
    return move.uci()


@app.post("/api/ai-step")
def post_ai_step(force: bool = Query(False)) -> Dict:
    """Advance AI by one move.
    - For Human vs AI (force=False): only when it's AI's turn.
    - For AI vs AI (force=True): move regardless of turn.
    This endpoint is intended for training/AI-vs-AI tools; normal play does not call it.
    """
    if board.is_game_over():
        return get_state()
    if not force and is_human_turn():
        raise HTTPException(status_code=400, detail="It's human's turn; set force=true for AI vs AI")
    ai_move()
    return get_state()


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

# Load weights on startup
load_ai_weights()

# -----------------------------
# Optional: Background training
# -----------------------------

training_thread: Optional[threading.Thread] = None
training_stop_flag = threading.Event()
training_status: Dict[str, object] = {
    "running": False,
    "iteration": 0,
    "accepted": 0,
}

# Async AI move runner (restored)
ai_thread: Optional[threading.Thread] = None
ai_thinking_flag = threading.Event()


def trigger_ai_move_async() -> None:
    global ai_thread
    if ai_thread and ai_thread.is_alive():
        return

    def _run():
        try:
            ai_thinking_flag.set()
            if not board.is_game_over():
                ai_move()
        finally:
            ai_thinking_flag.clear()
    ai_thread = threading.Thread(target=_run, daemon=True)
    ai_thread.start()


def play_self_play_game(weights_a: Dict[str, float], weights_b: Dict[str, float], max_time: float = 0.1) -> int:
    # Returns 1 if A wins, -1 if B wins, 0 draw
    b = chess.Board()
    # Alternate moves: A plays White, B plays Black
    # Temporarily swap global weights for search; restore after
    global AI_WEIGHTS
    original = AI_WEIGHTS.copy()
    try:
        while not b.is_game_over() and b.fullmove_number <= 80:
            if b.turn == chess.WHITE:
                AI_WEIGHTS = weights_a
            else:
                AI_WEIGHTS = weights_b
            move = search_best_move(b, "medium")
            if move is None:
                break
            b.push(move)
        result = b.result(claim_draw=True)
        if result == "1-0":
            return 1
        if result == "0-1":
            return -1
        return 0
    finally:
        AI_WEIGHTS = original


def training_worker(iterations: int = 20, games_per_iter: int = 4, sigma: float = 5.0) -> None:
    global training_status, training_thread
    training_status.update({"running": True, "iteration": 0, "accepted": 0})
    base = AI_WEIGHTS.copy()
    for it in range(iterations):
        if training_stop_flag.is_set():
            break
        training_status["iteration"] = it + 1
        # Propose candidate by Gaussian noise on numeric weights
        candidate = {k: float(v) for k, v in base.items()}
        for k in candidate:
            candidate[k] = candidate[k] + random.gauss(0.0, sigma)
        # Evaluate candidate vs base across multiple paired games (swap colors)
        score = 0
        for _ in range(max(1, games_per_iter)):
            # Candidate as White, Base as Black
            score += play_self_play_game(candidate, base)
            # Base as White, Candidate as Black (subtract, so positive favors candidate)
            score -= play_self_play_game(base, candidate)
        # If better or equal, accept
        if score > 0:
            base = candidate
            training_status["accepted"] = int(training_status.get("accepted", 0)) + 1
            AI_WEIGHTS.update(base)
            save_ai_weights()
    training_status["running"] = False
    training_thread = None


@app.post("/api/train-start")
def train_start(req: TrainStartRequest) -> Dict:
    global training_thread
    if training_thread and training_thread.is_alive():
        raise HTTPException(status_code=400, detail="Training already running")
    training_stop_flag.clear()
    training_thread = threading.Thread(target=training_worker, kwargs={
        "iterations": int(req.iterations or 20),
        "games_per_iter": int(req.games_per_iter or 4),
        "sigma": float(req.sigma or 5.0),
    }, daemon=True)
    training_thread.start()
    return {"status": "started", "iterations": int(req.iterations or 20)}


@app.get("/api/train-status")
def train_status() -> Dict:
    return {
        "running": training_status.get("running", False),
        "iteration": training_status.get("iteration", 0),
        "accepted": training_status.get("accepted", 0),
    }


@app.post("/api/train-stop")
def train_stop() -> Dict:
    training_stop_flag.set()
    return {"status": "stopping"}
