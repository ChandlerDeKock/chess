const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const newGameBtn = document.getElementById('newGameBtn');
const undoBtn = document.getElementById('undoBtn');
const flipBoardChk = document.getElementById('flipBoard');
const sizeSlider = document.getElementById('sizeSlider');
const sizeValue = document.getElementById('sizeValue');
const promotionModal = document.getElementById('promotionModal');

const PIECE_IMAGE = {
  'P': '/pieces/wP.svg', 'N': '/pieces/wN.svg', 'B': '/pieces/wB.svg', 'R': '/pieces/wR.svg', 'Q': '/pieces/wQ.svg', 'K': '/pieces/wK.svg',
  'p': '/pieces/bP.svg', 'n': '/pieces/bN.svg', 'b': '/pieces/bB.svg', 'r': '/pieces/bR.svg', 'q': '/pieces/bQ.svg', 'k': '/pieces/bK.svg',
};

let gameState = null;
let selectedSquare = null;
let legalTargets = [];
let flip = false;
let pendingPromotion = null; // { from, to }

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
  renderStatus();
  renderBoard();
}

function renderStatus() {
  if (!gameState) return;
  if (gameState.is_checkmate) {
    statusEl.textContent = `Checkmate. Winner: ${gameState.turn === 'white' ? 'black' : 'white'}`;
  } else if (gameState.is_stalemate) {
    statusEl.textContent = 'Stalemate.';
  } else if (gameState.is_check) {
    statusEl.textContent = `${gameState.turn} to move — Check!`;
  } else {
    statusEl.textContent = `${gameState.turn} to move.`;
  }
}

function pieceAt(square) {
  if (!gameState) return null;
  return gameState.pieces.find(p => p.square === square) || null;
}

function isOwnPiece(square) {
  const p = pieceAt(square);
  return p && p.color === gameState.turn;
}

async function onSquareClick(square) {
  if (pendingPromotion) return; // Wait for promotion selection

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

  // Attempt move selectedSquare -> square
  await tryMove(selectedSquare, square);
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
      div.className = `square ${squareColor(file, r-1)}${selectedSquare === sq ? ' selected' : ''}`;
      div.dataset.square = sq;

      const p = pieceAt(sq);
      if (p) {
        const img = document.createElement('img');
        img.className = 'piece';
        img.draggable = false;
        img.alt = p.symbol;
        img.src = PIECE_IMAGE[p.symbol];
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
  const res = await fetch('/api/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ from_square: fromSquare, to_square: toSquare, promotion }),
  });

  if (res.ok) {
    selectedSquare = null;
    legalTargets = [];
    gameState = await res.json();
    renderStatus();
    renderBoard();
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
  renderBoard();
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

flipBoardChk.addEventListener('change', () => {
  flip = !!flipBoardChk.checked;
  renderBoard();
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
