"""Rescore GSM8K answers with one final-answer extractor for every arm (lm_eval's flexible-extract takes the LAST
number in the text, which misreads closing sentences like "Claire will eat 7 dozens in 4 weeks").
usage: rescore.py RESULTS_DIR ARM [ARM ...]   (ARM = a results/<arm>/ folder holding samples_*.jsonl)

Extractor, first match wins:
  1. the last \\boxed{...} in the text -> the number in it
  2. the last "answer is / answer: / final answer ... X" phrase
  3. in the last sentence that contains a number: the first number after a verb-like link
     ("is", "are", "=", "equals", "makes", "earns", "needs", "will ...", "costs", "has", "gets", "spent", "takes", "pays")
  4. the last number in the text (the lm_eval flexible-extract behaviour)
Gold = the number after '####'. Numbers compared as floats after removing ',', '$' and a trailing '.'.
Prints per arm: lm_eval flexible, rescored, and which rule fired; then paired counts between the first two arms.
"""
import glob, json, re, sys

NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def norm(x):
    try:
        return float(x.replace(",", "").replace("$", "").rstrip("."))
    except ValueError:
        return None


def nums(s):
    return re.findall(NUM, s)


def extract(text):
    t = text.strip()
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", t)
    if boxed:
        n = nums(boxed[-1])
        if n:
            return n[-1], "boxed"
    m = re.findall(r"(?:final answer|answer)\s*(?:is|:|=)\s*[:\s]*\**\s*(" + NUM + ")", t, re.I)
    if m:
        return m[-1], "answer-is"
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", t) if nums(s)]
    if sentences:
        last = sentences[-1]
        m = re.search(r"\b(?:is|are|=|equals|makes|earns|needs|costs|has|gets|spent|spends|takes|pays|will \w+|would \w+)"
                      r"\s+(?:a total of\s+|about\s+|approximately\s+|\\\(\s*)?(" + NUM + ")", last)
        if m:
            return m.group(1), "last-sentence"
    n = nums(t)
    return (n[-1], "last-number") if n else (None, "none")


def load(results, arm):
    path = glob.glob(f"{results}/{arm}/**/samples_*.jsonl", recursive=True)
    assert len(path) == 1, (arm, path)
    out = {}
    for line in open(path[0]):
        d = json.loads(line)
        if d.get("filter") != "flexible-extract":
            continue
        gold = norm(d["target"].split("####")[-1].strip())
        ans, rule = extract(d["resps"][0][0])
        out[d["doc_id"]] = {"lm_eval": bool(d["exact_match"]), "ok": ans is not None and norm(ans) == gold,
                            "rule": rule, "ans": ans, "gold": gold, "text": d["resps"][0][0]}
    return out


def main():
    results, arms = sys.argv[1], sys.argv[2:]
    data = {a: load(results, a) for a in arms}
    try:  # hand-checked corrections (see hand_checked.json)
        hc = json.load(open("hand_checked.json"))
    except FileNotFoundError:
        hc = {}
    for a, rows in data.items():
        for i in hc.get(a, {}).get("correct", []):
            rows[i]["ok"], rows[i]["rule"] = True, "hand-checked"
    for a, rows in data.items():
        rules = {}
        for r in rows.values():
            rules[r["rule"]] = rules.get(r["rule"], 0) + 1
        flipped_up = sum(1 for r in rows.values() if r["ok"] and not r["lm_eval"])
        flipped_down = sum(1 for r in rows.values() if r["lm_eval"] and not r["ok"])
        print(f"{a}: n={len(rows)} lm_eval={sum(r['lm_eval'] for r in rows.values())} "
              f"rescored={sum(r['ok'] for r in rows.values())} (+{flipped_up} / -{flipped_down}) rules={rules}")
    if len(arms) >= 2:
        a, b = data[arms[0]], data[arms[1]]
        ids = sorted(set(a) & set(b))
        only_a = sum(1 for i in ids if a[i]["ok"] and not b[i]["ok"])
        only_b = sum(1 for i in ids if b[i]["ok"] and not a[i]["ok"])
        print(f"paired (rescored) {arms[0]} right alone: {only_a}; {arms[1]} right alone: {only_b}; both: "
              f"{sum(1 for i in ids if a[i]['ok'] and b[i]['ok'])}")


if __name__ == "__main__":
    main()
