#!/usr/bin/env python3
"""Write the eight judge output schemas byte-exactly from the one-line *.min.json beside this file, and check each
file's sha256 against the digest the judge prompts pinned. Exit 1 on any mismatch.

  python materialize.py [--out DIR]          (default: this directory)

Six are re-rendered with the layout their runs used (json.dumps indent=1, ensure_ascii=False; the harbor schemas also
sort_keys=True and end with a newline). The two *_k3 schemas are their base with reason.maxLength 600 -> 4000, the only
change, made for the Kimi K3 runs.
"""
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# name: (sort_keys, trailing newline, sha256)
RENDERED = {
    "output_schema_v13.json": (False, "", "f8a47506d0d060c12d149e427d59f3b211f03e56cb20bcfb9b74e0c8a3d2db1e"),
    "output_schema_v13_rst.json": (False, "", "1cf0bd9b7a50c05d7b77ba94cfa0a834cdfbdd7a73b917b5a019c3e198fb1ef2"),
    "output_schema_v13_harbor21.json": (True, "\n", "0d37ee14631bc41531df0af8b15e29d387a33c95e80d713a73fd5cb3d2b439ad"),
    "output_schema_v13_harbor3.json": (True, "\n", "d8d695a62d8ae8e39ab4645b9e9fe8aa57153c83b4027bfa85c89117d0edb9dd"),
    "output_schema_v13_harbor4xm.json": (True, "\n", "68e1f99599a93a5ef06db35d4f80fa48e5e4ae283059e59e338e7f97e2ba889b"),
    "output_schema_v13_harbor4xm_c.json": (True, "\n", "bec8bee2b791cb3e9e4953466d585d2ca5ac020db69771e743b4bc5d02901edc"),
}
# name: (base, 1-based line holding reason.maxLength, sha256)
K3 = {
    "output_schema_v13_k3.json": ("output_schema_v13.json", 278,
                                  "0af911a6454cacda8ea9ddff81263de1216f821bb3398f5c5e0911ca0ccc4d1c"),
    "output_schema_v13_harbor3_k3.json": ("output_schema_v13_harbor3.json", 1787,
                                          "d33a570f5d17e095cdd0682b36cc906700ea5f19a12f12b8156d56834ebcbd58"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE)
    out_dir = ap.parse_args().out
    text, want = {}, {}
    for name, (sort_keys, tail, sha) in RENDERED.items():
        s = json.load(open(os.path.join(HERE, name.replace(".json", ".min.json")), encoding="utf-8"))
        text[name] = json.dumps(s, indent=1, sort_keys=sort_keys, ensure_ascii=False) + tail
        want[name] = sha
    for name, (base, line_no, sha) in K3.items():
        lines = text[base].split("\n")
        assert lines[line_no - 1].strip().startswith('"maxLength": 600'), (name, lines[line_no - 1])
        lines[line_no - 1] = lines[line_no - 1].replace("600", "4000")
        text[name], want[name] = "\n".join(lines), sha
    bad = 0
    for name, t in text.items():
        h = hashlib.sha256(t.encode("utf-8")).hexdigest()
        bad += h != want[name]
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(t)
        print("%s %s %s" % ("ok " if h == want[name] else "BAD", h[:8], name))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
