"""Compositional SAN tokenizer/detokenizer for chess PGN movetext.

This is a notation codec, not a legality checker. It converts SAN moves like
``Nbd7`` into expressive tokens and stable integer IDs, and can reconstruct the
SAN string from those tokens/IDs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

FILES = "abcdefgh"
RANKS = "12345678"
PIECES = "KQRBN"
PROMOS = "QRBN"

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<GAME_START>", "<GAME_END>", "<UNK>", "<EOM>"]
TURN_TOKENS = ["WHITE", "BLACK"]
PIECE_TOKENS = ["PAWN", *[f"PIECE_{p}" for p in PIECES]]
SRC_FILE_TOKENS = [f"SRC_FILE_{f}" for f in FILES]
SRC_RANK_TOKENS = [f"SRC_RANK_{r}" for r in RANKS]
TO_TOKENS = [f"TO_{f}{r}" for r in RANKS for f in FILES]
ACTION_TOKENS = ["CAPTURE", "CHECK", "MATE"]
CASTLE_TOKENS = ["CASTLE_KINGSIDE", "CASTLE_QUEENSIDE"]
PROMO_TOKENS = [f"PROMO_{p}" for p in PROMOS]
RESULT_TOKENS = ["RESULT_WHITE", "RESULT_BLACK", "RESULT_DRAW", "RESULT_UNKNOWN"]

VOCAB: list[str] = (
    SPECIAL_TOKENS
    + TURN_TOKENS
    + PIECE_TOKENS
    + SRC_FILE_TOKENS
    + SRC_RANK_TOKENS
    + TO_TOKENS
    + ACTION_TOKENS
    + CASTLE_TOKENS
    + PROMO_TOKENS
    + RESULT_TOKENS
)
TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}

RESULT_TO_TOKEN = {"1-0": "RESULT_WHITE", "0-1": "RESULT_BLACK", "1/2-1/2": "RESULT_DRAW", "*": "RESULT_UNKNOWN"}
TOKEN_TO_RESULT = {v: k for k, v in RESULT_TO_TOKEN.items()}

COMMENT_RE = re.compile(r"\{[^}]*\}|;[^\n]*")
NAG_RE = re.compile(r"\$\d+")
MOVE_NUMBER_RE = re.compile(r"^\d+\.(?:\.\.)?$")
SAN_RE = re.compile(
    r"^"
    r"(?P<piece>[KQRBN])?"
    r"(?P<src_file>[a-h])?"
    r"(?P<src_rank>[1-8])?"
    r"(?P<capture>x)?"
    r"(?P<target>[a-h][1-8])"
    r"(?:=(?P<promo>[QRBN]))?"
    r"(?P<suffix>[+#])?"
    r"(?P<annotation>[!?]+)?"
    r"$"
)


@dataclass(frozen=True)
class EncodedMove:
    san: str
    tokens: list[str]
    ids: list[int]


class ChessTokenizer:
    """Chess-aware SAN tokenizer with stable token IDs."""

    vocab = VOCAB
    token_to_id = TOKEN_TO_ID
    id_to_token = ID_TO_TOKEN

    def token_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id["<UNK>"])

    def encode_tokens(self, tokens: Sequence[str]) -> list[int]:
        return [self.token_id(token) for token in tokens]

    def decode_ids(self, ids: Sequence[int]) -> list[str]:
        return [self.id_to_token.get(int(idx), "<UNK>") for idx in ids]

    def encode_move(self, san: str) -> EncodedMove:
        tokens = tokenize_san(san)
        return EncodedMove(san=san, tokens=tokens, ids=self.encode_tokens(tokens))

    def decode_move(self, ids_or_tokens: Sequence[int | str]) -> str:
        if not ids_or_tokens:
            return ""
        if isinstance(ids_or_tokens[0], int):
            tokens = self.decode_ids(ids_or_tokens)  # type: ignore[arg-type]
        else:
            tokens = list(ids_or_tokens)  # type: ignore[assignment]
        return detokenize_san(tokens)

    def tokenize_movetext(self, movetext: str, include_turn_tokens: bool = True) -> list[str]:
        return tokenize_movetext(movetext, include_turn_tokens=include_turn_tokens)

    def encode_movetext(self, movetext: str, include_turn_tokens: bool = True) -> list[int]:
        return self.encode_tokens(self.tokenize_movetext(movetext, include_turn_tokens=include_turn_tokens))

    def detokenize_movetext(self, ids_or_tokens: Sequence[int | str]) -> str:
        if not ids_or_tokens:
            return ""
        if isinstance(ids_or_tokens[0], int):
            tokens = self.decode_ids(ids_or_tokens)  # type: ignore[arg-type]
        else:
            tokens = list(ids_or_tokens)  # type: ignore[assignment]
        return detokenize_movetext(tokens)


def strip_variations(text: str) -> str:
    out: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
            continue
        if ch == ")" and depth:
            depth -= 1
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


def clean_movetext(text: str) -> str:
    text = COMMENT_RE.sub(" ", text)
    text = strip_variations(text)
    text = NAG_RE.sub(" ", text)
    return text.replace("\n", " ")


def pgn_to_movetext(pgn: str) -> str:
    return "\n".join(line for line in pgn.splitlines() if not line.startswith("["))


def normalize_castle(san: str) -> str:
    if san in {"0-O", "0-O-O", "0-0", "0-0-0"}:
        return san.replace("0", "O")
    return san


def tokenize_san(san: str, add_eom: bool = True) -> list[str]:
    san = san.strip()
    if not san:
        return []
    san = normalize_castle(san)

    if san in RESULT_TO_TOKEN:
        return [RESULT_TO_TOKEN[san]]

    if san.startswith("O-O-O"):
        tokens = ["CASTLE_QUEENSIDE"]
        suffix = san[5:]
    elif san.startswith("O-O"):
        tokens = ["CASTLE_KINGSIDE"]
        suffix = san[3:]
    else:
        san_core = re.sub(r"[!?]+$", "", san)
        match = SAN_RE.match(san_core)
        if not match:
            return ["<UNK>", "<EOM>"] if add_eom else ["<UNK>"]

        gd = match.groupdict()
        piece = gd["piece"] or "P"
        tokens = ["PAWN" if piece == "P" else f"PIECE_{piece}"]
        if gd["src_file"]:
            tokens.append(f"SRC_FILE_{gd['src_file']}")
        if gd["src_rank"]:
            tokens.append(f"SRC_RANK_{gd['src_rank']}")
        if gd["capture"]:
            tokens.append("CAPTURE")
        tokens.append(f"TO_{gd['target']}")
        if gd["promo"]:
            tokens.append(f"PROMO_{gd['promo']}")
        suffix = gd["suffix"] or ""

    if "+" in suffix:
        tokens.append("CHECK")
    if "#" in suffix:
        tokens.append("MATE")
    if add_eom:
        tokens.append("<EOM>")
    return tokens


def tokenize_movetext(movetext: str, include_turn_tokens: bool = True) -> list[str]:
    cleaned = clean_movetext(movetext)
    tokens: list[str] = []
    ply = 0

    for raw in cleaned.split():
        explicit_black_move = bool(re.match(r"^\d+\.\.\.", raw))
        raw = re.sub(r"^\d+\.{1,3}", "", raw.strip())
        if not raw or MOVE_NUMBER_RE.match(raw):
            continue
        if explicit_black_move and ply % 2 == 0:
            ply += 1

        move_tokens = tokenize_san(raw)
        if include_turn_tokens and move_tokens and not move_tokens[0].startswith("RESULT_"):
            tokens.append("WHITE" if ply % 2 == 0 else "BLACK")
            ply += 1
        tokens.extend(move_tokens)
        if move_tokens and move_tokens[0].startswith("RESULT_"):
            break
    return tokens


def detokenize_san(tokens: Sequence[str]) -> str:
    """Reconstruct one SAN move from tokens for a single move.

    Turn tokens and EOM are tolerated and ignored. Result tokens return their PGN
    result string. The output is canonical SAN without annotation glyphs (!?).
    """
    toks = [t for t in tokens if t not in {"WHITE", "BLACK", "<EOM>", "<BOS>", "<EOS>", "<PAD>"}]
    if not toks:
        return ""
    if toks[0] in TOKEN_TO_RESULT:
        return TOKEN_TO_RESULT[toks[0]]
    if toks[0] == "CASTLE_KINGSIDE":
        san = "O-O"
    elif toks[0] == "CASTLE_QUEENSIDE":
        san = "O-O-O"
    else:
        piece = ""
        src_file = ""
        src_rank = ""
        capture = ""
        target = ""
        promo = ""
        for tok in toks:
            if tok == "PAWN":
                piece = ""
            elif tok.startswith("PIECE_"):
                piece = tok.split("_", 1)[1]
            elif tok.startswith("SRC_FILE_"):
                src_file = tok[-1]
            elif tok.startswith("SRC_RANK_"):
                src_rank = tok[-1]
            elif tok == "CAPTURE":
                capture = "x"
            elif tok.startswith("TO_"):
                target = tok[3:]
            elif tok.startswith("PROMO_"):
                promo = "=" + tok[-1]
        if not target:
            return "?"
        san = f"{piece}{src_file}{src_rank}{capture}{target}{promo}"

    if "MATE" in toks:
        san += "#"
    elif "CHECK" in toks:
        san += "+"
    return san


def split_token_moves(tokens: Sequence[str]) -> Iterable[list[str]]:
    current: list[str] = []
    for tok in tokens:
        if tok in {"WHITE", "BLACK"} and not current:
            current.append(tok)
            continue
        current.append(tok)
        if tok == "<EOM>" or tok.startswith("RESULT_"):
            yield current
            current = []
    if current:
        yield current


def detokenize_movetext(tokens: Sequence[str]) -> str:
    moves: list[str] = []
    result = ""
    ply = 0
    for move_tokens in split_token_moves(tokens):
        san = detokenize_san(move_tokens)
        if not san:
            continue
        if san in RESULT_TO_TOKEN:
            result = san
            break
        if ply % 2 == 0:
            moves.append(f"{ply // 2 + 1}. {san}")
        else:
            moves[-1] += f" {san}"
        ply += 1
    if result:
        moves.append(result)
    return " ".join(moves)


DEFAULT_TOKENIZER = ChessTokenizer()
