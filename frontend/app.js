const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const newGameBtn = document.getElementById('newGameBtn');
const undoBtn = document.getElementById('undoBtn');
const flipBoardChk = document.getElementById('flipBoard');
const colorSelect = document.getElementById('colorSelect');
const difficultySelect = document.getElementById('difficultySelect');
const sizeSlider = document.getElementById('sizeSlider');
const sizeValue = document.getElementById('sizeValue');
const promotionModal = document.getElementById('promotionModal');

// Use local SVGs for standard-looking pieces
const PIECE_IMAGE = {
  'P': '/pieces/wP.svg', 'N': '/pieces/wN.svg', 'B': '/pieces/wB.svg', 'R': '/pieces/wR.svg', 'Q': '/pieces/wQ.svg', 'K': '/pieces/wK.svg',
  'p': '/pieces/bP.svg', 'n': '/pieces/bN.svg', 'b': '/pieces/bB.svg', 'r': '/pieces/bR.svg', 'q': '/pieces/bQ.svg', 'k': '/pieces/bK.svg',
};

let gameState = null;
let selectedSquare = null;
let legalTargets = [];
let flip = false;
let pendingPromotion = null; // { from, to }
let lastMoveFrom = null;
let lastMoveTo = null;
let aiLoop = null; // reserved for training/ai-vs-ai flows (not used in human vs AI)
let moveInFlight = false; // prevent duplicate move submissions

function algebraicToCoords(square) {
  const file = square.charCodeAt(0) - 'a'.charCodeAt(0); // 0..7
  const rank = parseInt(square[1], 10) - 1; // 0..7
  return { file, rank };
}

function coordsToIndex(file, rank) {
  return rank * 8 + file;
}

function indexToSquare(index) {
  const file = index % 8;
  const rank = Math.floor(index / 8);
  return String.fromCharCode('a'.charCodeAt(0) + file) + (rank + 1);
}

function squareColor(file, rank) {
  return (file + rank) % 2 === 0 ? 'light' : 'dark';
}

async function fetchState() {
  const res = await fetch('/api/state');
  gameState = await res.json();
  // Sync selectors if changed elsewhere
  if (colorSelect && gameState.human_color && colorSelect.value !== gameState.human_color) {
    colorSelect.value = gameState.human_color;
  }
  if (difficultySelect && gameState.ai_difficulty && difficultySelect.value !== gameState.ai_difficulty) {
    difficultySelect.value = gameState.ai_difficulty;
  }
  // Update last move markers from history if available
  if (gameState && Array.isArray(gameState.history) && gameState.history.length > 0) {
    const lastUci = gameState.history[gameState.history.length - 1];
    // UCI: e2e4 or e7e8q (promotion suffix ignored for highlights)
    if (typeof lastUci === 'string' && lastUci.length >= 4) {
      lastMoveFrom = lastUci.slice(0, 2);
      lastMoveTo = lastUci.slice(2, 4);
    }
  } else {
    lastMoveFrom = null;
    lastMoveTo = null;
  }
  renderStatus();
  renderBoard();
  // In Human vs AI, no autonomous loop; server replies synchronously for AI turns
}

function renderStatus() {
  if (!gameState) return;
  const yourTurn = (gameState.turn === gameState.human_color);
  if (gameState.is_checkmate) {
    statusEl.textContent = `Checkmate. Winner: ${gameState.turn === 'white' ? 'black' : 'white'}`;
  } else if (gameState.is_stalemate) {
    statusEl.textContent = 'Stalemate.';
  } else if (gameState.is_check) {
    statusEl.textContent = `${gameState.turn} to move — Check!${yourTurn ? '' : (gameState.ai_thinking ? ' (AI thinking...)' : '')}`;
  } else {
    statusEl.textContent = `${gameState.turn} to move.${yourTurn ? '' : (gameState.ai_thinking ? ' (AI thinking...)' : '')}`;
  }
}

function pieceAt(square) {
  if (!gameState) return null;
  return gameState.pieces.find(p => p.square === square) || null;
}

function isOwnPiece(square) {
  const p = pieceAt(square);
  return p && p.color === gameState.human_color;
}

async function onSquareClick(square) {
  if (pendingPromotion) return; // Wait for promotion selection
  if (!gameState || gameState.turn !== gameState.human_color) return; // Only allow moves on your turn
  if (moveInFlight) return; // Block input while move is processing

  if (selectedSquare === null) {
    if (isOwnPiece(square)) {
      selectedSquare = square;
      legalTargets = await getLegalTargets(square);
      renderBoard();
    }
    return;
  }

  if (square === selectedSquare) {
    selectedSquare = null;
    legalTargets = [];
    renderBoard();
    return;
  }

  // Only allow moves to legal targets; allow reselecting another own piece
  if (legalTargets.includes(square)) {
    await tryMove(selectedSquare, square);
  } else if (isOwnPiece(square)) {
    selectedSquare = square;
    legalTargets = await getLegalTargets(square);
    renderBoard();
  } else {
    // Ignore invalid target; keep current selection
    return;
  }
}

async function getLegalTargets(fromSquare) {
  const res = await fetch(`/api/legal-moves?from=${fromSquare}`);
  const data = await res.json();
  return data.targets;
}

function shouldShowCaptureRing(square) {
  const p = pieceAt(square);
  return !!p && legalTargets.includes(square) && selectedSquare !== null;
}

function shouldShowDot(square) {
  return !pieceAt(square) && legalTargets.includes(square) && selectedSquare !== null;
}

function renderBoard() {
  boardEl.innerHTML = '';

  // We draw ranks 8..1 if not flipped; reversed when flipped.
  const ranks = flip ? [1,2,3,4,5,6,7,8] : [8,7,6,5,4,3,2,1];
  const files = flip ? ['h','g','f','e','d','c','b','a'] : ['a','b','c','d','e','f','g','h'];

  for (let r of ranks) {
    for (let f of files) {
      const sq = `${f}${r}`;
      const { file, rank } = algebraicToCoords(sq);
      const div = document.createElement('div');
      const isSelected = selectedSquare === sq;
      const isLastFrom = lastMoveFrom === sq;
      const isLastTo = lastMoveTo === sq;
      div.className = `square ${squareColor(file, r-1)}${isSelected ? ' selected' : ''}${isLastFrom ? ' last-from' : ''}${isLastTo ? ' last-to' : ''}`;
      div.dataset.square = sq;

      const p = pieceAt(sq);
      if (p) {
        const img = document.createElement('img');
        img.className = 'piece';
        img.draggable = false;
        img.alt = p.symbol;
        img.src = PIECE_IMAGE[p.symbol];
        if (lastMoveTo === sq) {
          img.classList.add('moved');
        }
        div.appendChild(img);
      }

      if (shouldShowDot(sq)) {
        const dot = document.createElement('div');
        dot.className = 'dot';
        div.appendChild(dot);
      }
      if (shouldShowCaptureRing(sq)) {
        const ring = document.createElement('div');
        ring.className = 'capture';
        div.appendChild(ring);
      }

      div.addEventListener('click', () => onSquareClick(sq));
      boardEl.appendChild(div);
    }
  }
}

async function tryMove(fromSquare, toSquare, promotion = null) {
  moveInFlight = true;
  let res;
  try {
    res = await fetch('/api/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ from_square: fromSquare, to_square: toSquare, promotion }),
    });
  } catch (e) {
    // Network or server error; reset selection and refresh
    selectedSquare = null;
    legalTargets = [];
    await fetchState();
    moveInFlight = false;
    return;
  }

  if (res.ok) {
    selectedSquare = null;
    legalTargets = [];
    gameState = await res.json();
    renderStatus();
    renderBoard();
    // As before, refresh once shortly after to ensure we reflect any server-side AI reply
    setTimeout(fetchState, 50);
    moveInFlight = false;
    return;
  }

  // 409 for promotion required
  if (res.status === 409) {
    let data = null;
    try { data = await res.json(); } catch {}
    if (data && data.requires_promotion) {
      pendingPromotion = { from: fromSquare, to: toSquare };
      openPromotionModal();
      return;
    }
  }

  // Otherwise, invalid move
  selectedSquare = null;
  legalTargets = [];
  await fetchState();
  moveInFlight = false;
}

function openPromotionModal() {
  promotionModal.classList.remove('hidden');
}

function closePromotionModal() {
  promotionModal.classList.add('hidden');
}

promotionModal.addEventListener('click', (e) => {
  if (e.target === promotionModal) {
    pendingPromotion = null;
    closePromotionModal();
  }
});

promotionModal.querySelectorAll('button[data-piece]').forEach(btn => {
  btn.addEventListener('click', async () => {
    if (!pendingPromotion) return;
    const piece = btn.getAttribute('data-piece');
    await tryMove(pendingPromotion.from, pendingPromotion.to, piece);
    pendingPromotion = null;
    closePromotionModal();
  });
});

newGameBtn.addEventListener('click', async () => {
  await fetch('/api/new-game', { method: 'POST' });
  await fetchState();
});

undoBtn.addEventListener('click', async () => {
  await fetch('/api/undo', { method: 'POST' });
  await fetchState();
});

// No mode selector in Human vs AI mode

flipBoardChk.addEventListener('change', () => {
  flip = !!flipBoardChk.checked;
  renderBoard();
});

colorSelect.addEventListener('change', async () => {
  const val = colorSelect.value;
  await fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ human_color: val }),
  });
  // Start a new game automatically when color changes
  await fetch('/api/new-game', { method: 'POST' });
  await fetchState();
});

difficultySelect.addEventListener('change', async () => {
  const val = difficultySelect.value;
  await fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ difficulty: val }),
  });
  // Keep current game; difficulty applies on AI's next move
  await fetchState();
});

function updateBoardSize(px) {
  document.documentElement.style.setProperty('--board-size', `${px}px`);
  sizeValue.textContent = `${px}px`;
}

sizeSlider.addEventListener('input', (e) => {
  const val = parseInt(e.target.value, 10);
  updateBoardSize(val);
});

// Initialize size to slider value
updateBoardSize(parseInt(sizeSlider.value, 10));

fetchState();

// AI vs AI logic moved out of app.js (used only by training panel)
