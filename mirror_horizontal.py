"""Horizontal (file-axis) mirror augmentation: a<->h, b<->g, c<->f, d<->e. This is
a genuine symmetry of chess (unlike an arbitrary transform) -- a mirrored position
has identical game-theoretic value to the original, so the cp label carries over
unchanged. Doubles a dataset's effective size for free.

Per rank string, simple character-reversal correctly mirrors the files: each
character (a piece letter or a digit run-length 1-8) represents exactly one
square-run in file order, so reversing the character sequence reverses the file
order too -- no special-casing needed for multi-square digit runs.

Castling rights are the reason this ISN'T always valid: king and queen are not
mirror-symmetric pieces -- they sit on fixed, different home files (e and d), not
a matching pair like the two knights/bishops/rooks. Mirroring a position that
still has live castling rights moves the king off e1/e8, which makes standard
(non-Chess960) castling notation self-contradictory -- python-chess itself
rejects it as STATUS_BAD_CASTLING_RIGHTS (confirmed directly: mirroring the
starting position produces exactly that error). So this only mirrors positions
where castling is already "-" for both sides; anything else is skipped rather
than produce a technically-invalid FEN. Real games lose all rights well before
most middlegame/endgame positions, so this still keeps a large majority of rows.

En passant square's file mirrors the same way as a board square; its rank is
unchanged. Side to move, halfmove clock, and fullmove number are untouched by a
pure left-right reflection.

Writes ONLY the mirrored rows (same count as input, minus skipped live-castling
ones) -- concatenate with the original file yourself to actually double the
training set, so this stays reusable for "just show me the mirrored version"
checks too.

    uv run python mirror_horizontal.py data/raw_quiet.jsonl data/raw_quiet_mirrored.jsonl
"""
import json
import sys

FILE_MIRROR = str.maketrans("abcdefgh", "hgfedcba")


def mirror_fen(fen: str) -> str | None:
    fields = fen.split(" ")
    board, turn, castling, ep = fields[0], fields[1], fields[2], fields[3]
    rest = fields[4:]

    if castling != "-":
        return None

    mirrored_board = "/".join(rank[::-1] for rank in board.split("/"))
    if ep == "-":
        mirrored_ep = "-"
    else:
        mirrored_ep = ep[0].translate(FILE_MIRROR) + ep[1:]

    return " ".join([mirrored_board, turn, castling, mirrored_ep, *rest])


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    written = 0
    skipped_castling = 0
    total = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                data = json.loads(line)
                mirrored_fen = mirror_fen(data["fen"])
            except (json.JSONDecodeError, KeyError, ValueError, IndexError):
                continue
            if mirrored_fen is None:
                skipped_castling += 1
                continue
            fout.write(json.dumps({"fen": mirrored_fen, "cp": data["cp"]}) + "\n")
            written += 1
    print(f"wrote {written:,} of {total:,} rows to {out_path} "
          f"({skipped_castling:,} skipped: still had live castling rights)", file=sys.stderr)


if __name__ == "__main__":
    main()
