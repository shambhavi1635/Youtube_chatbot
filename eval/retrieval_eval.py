"""Retrieval and groundedness evaluation for the YouTube RAG pipeline.

Measures the pipeline that is already in `backend/pipeline.py` -- it imports
`split_transcript`, `build_vector_store`, `build_retriever` and `build_chain`
rather than reimplementing any of them, and it changes no tuning constant.

Three things are measured:

1. Retrieval hit rate at k = 1, 4, 8. A question counts as a hit at k if any of
   the top-k retrieved chunks contains that question's gold substring (an exact
   phrase copied by hand out of the transcript passage that answers it).
2. Answer groundedness. The full chain is run on each question and the answer is
   classified correct / incorrect / abstained. The prompt in pipeline.py tells the
   model to say it does not know when the context is insufficient, so abstentions
   are counted separately instead of being folded into "incorrect".
3. Latency. Indexing time per video and query time per question, reported as
   median and p90.

Usage
-----
    # No API key needed: checks that every gold substring really is in a chunk.
    python eval/retrieval_eval.py --validate-only

    # Retrieval metrics + latency, no chain calls (cheaper on quota).
    python eval/retrieval_eval.py --no-answers

    # Everything.
    python eval/retrieval_eval.py

Needs GOOGLE_API_KEY in .env (pipeline.load_dotenv picks it up) for anything
other than --validate-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from typing import Any

# Import the pipeline under test from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# pipeline.py calls load_dotenv() with no path, which resolves relative to the
# working directory -- so running this script from anywhere but the repo root
# would not find the key. Load the repo's own .env explicitly. This does not
# override anything already in the environment.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_ROOT, ".env"))

from backend.pipeline import (  # noqa: E402
    RETRIEVER_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    QuotaExceeded,
    _is_quota_error,
    answer_question,
    build_chain,
    build_retriever,
    build_vector_store,
    fetch_transcript,
    require_api_key,
    split_transcript,
)

DEFAULT_K_LIST = (1, RETRIEVER_K, 8)

# Phrases that mean the model declined to answer. Checked only after the keyword
# test fails, so a correct-but-hedged answer is still scored correct.
ABSTAIN_PATTERNS = (
    "i don't know",
    "i do not know",
    "i dont know",
    "don't know",
    "do not know",
    "cannot answer",
    "can't answer",
    "cannot determine",
    "unable to answer",
    "insufficient",
    "not enough information",
    "no information",
    "not mentioned",
    "not provided",
    "does not provide",
    "doesn't provide",
    "does not contain",
    "doesn't contain",
    "not specified",
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "for", "from",
    "has", "have", "in", "into", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "them", "there", "these", "they", "this", "to", "was",
    "were", "which", "with", "you", "your",
}


# ---------------------------------------------------------------- normalisation

def _norm(text: str) -> str:
    """Normalise for substring matching.

    Captions carry arbitrary line breaks inside a sentence, so whitespace is
    collapsed. Digit group separators are stripped so 13,000 and 13000 match.
    Curly apostrophes are folded to straight ones.
    """
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", _norm(text))
    return {w for w in words if w not in _STOPWORDS}


def _keyword_present(answer_norm: str, keyword: Any) -> bool:
    """A string keyword must appear; a list means any one alternative counts."""
    if isinstance(keyword, (list, tuple)):
        return any(_norm(alt) in answer_norm for alt in keyword)
    return _norm(keyword) in answer_norm


# -------------------------------------------------------------------- statistics

def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Honest for the small n we have here."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval -- the honest error bar on a 30-item dataset."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _summarise_times(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median_s": None, "p90_s": None, "min_s": None, "max_s": None}
    return {
        "n": len(values),
        "median_s": round(statistics.median(values), 3),
        "p90_s": round(_percentile(values, 90), 3),
        "min_s": round(min(values), 3),
        "max_s": round(max(values), 3),
    }


# ------------------------------------------------------------------ quota guard

class RateLimiter:
    """Keeps embedding/LLM calls under the Gemini free-tier per-minute ceiling.

    The free tier allows 100 embedding requests per minute. One indexing call
    embeds a whole video in batches, so this paces whole operations rather than
    individual HTTP requests, which is why the default budget sits well below 100.
    """

    def __init__(self, calls_per_minute: int):
        self.budget = max(1, calls_per_minute)
        self.stamps: list[float] = []

    def take(self, weight: int = 1) -> None:
        for _ in range(max(1, weight)):
            now = time.time()
            self.stamps = [t for t in self.stamps if now - t < 60.0]
            if len(self.stamps) >= self.budget:
                wait = 60.0 - (now - self.stamps[0]) + 0.5
                if wait > 0:
                    print(f"    [rate limit] sleeping {wait:.0f}s to stay under "
                          f"{self.budget} calls/min", flush=True)
                    time.sleep(wait)
                now = time.time()
                self.stamps = [t for t in self.stamps if now - t < 60.0]
            self.stamps.append(time.time())


_slept_total = 0.0


def reset_slept() -> None:
    """Start a fresh accounting window for backoff sleep."""
    global _slept_total
    _slept_total = 0.0


def slept_since_reset() -> float:
    """Seconds spent asleep waiting on quota since the last reset.

    Timed regions subtract this: a 65s backoff is time spent waiting for a free
    tier window to reopen, not time the pipeline spent indexing or answering.
    Counting it as latency would overstate the measurement by an order of
    magnitude on any call that happened to hit the quota.
    """
    return _slept_total


def with_quota_retry(fn, label: str, attempts: int = 5, base_sleep: float = 65.0):
    """Run fn, backing off on 429 / RESOURCE_EXHAUSTED.

    pipeline.QuotaExceeded already wraps those, and _is_quota_error catches the
    ones raised from code paths that do not wrap.
    """
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except QuotaExceeded as exc:
            last = exc
        except Exception as exc:  # noqa: BLE001 - only quota errors are retried
            if not _is_quota_error(exc):
                raise
            last = exc
        sleep_for = base_sleep * attempt
        if attempt == attempts:
            break
        print(f"    [quota] {label}: {type(last).__name__} on attempt "
              f"{attempt}/{attempts}; sleeping {sleep_for:.0f}s", flush=True)
        global _slept_total
        time.sleep(sleep_for)
        _slept_total += sleep_for
    raise QuotaExceeded(f"{label}: quota errors persisted after {attempts} attempts: {last}")


# ------------------------------------------------------------------- transcripts

def load_transcript(video_id: str, cache_dir: str | None) -> tuple[str, float, bool]:
    """Return (text, seconds_spent_fetching, came_from_cache).

    Uses the repo's own fetch_transcript so the indexed text is byte-for-byte
    what the app would index. Caching keeps repeat runs reproducible if YouTube
    captions change under us.
    """
    path = os.path.join(cache_dir, f"{video_id}.txt") if cache_dir else None
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read(), 0.0, True

    started = time.perf_counter()
    text = fetch_transcript(video_id)
    elapsed = time.perf_counter() - started

    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text, elapsed, False


# -------------------------------------------------------------------- validation

def validate_dataset(dataset: dict, cache_dir: str | None) -> int:
    """Check every gold substring is really present in some chunk.

    A gold substring that straddles a chunk boundary could never be retrieved by
    any k, which would be a dataset bug rather than a retrieval failure. This
    runs without an API key: it is pure text.
    """
    questions = dataset["questions"]
    by_video: dict[str, list[dict]] = {}
    for q in questions:
        by_video.setdefault(q["video_id"], []).append(q)

    problems = 0
    for video_id, qs in by_video.items():
        text, _, cached = load_transcript(video_id, cache_dir)
        chunks = split_transcript(text)
        normed = [_norm(c.page_content) for c in chunks]
        print(f"\n{video_id}: {len(text)} chars, {len(chunks)} chunks "
              f"({'cached' if cached else 'fetched'})")
        for q in qs:
            needle = _norm(q["gold_substring"])
            holders = [i for i, c in enumerate(normed) if needle in c]
            if holders:
                print(f"  ok    {q['id']}: chunk(s) {holders}")
            else:
                problems += 1
                print(f"  FAIL  {q['id']}: gold substring not found in any chunk")
                print(f"        {q['gold_substring'][:100]!r}")
    print(f"\n{len(questions)} questions checked, {problems} unmatched.")
    return problems


# ------------------------------------------------------------------- scoring

def score_answer(answer: str, question: dict) -> dict[str, Any]:
    """Classify an answer as correct / incorrect / abstained.

    Correctness is keyword presence, not an LLM judge: with 30 questions a judge
    would add cost and a second source of error for no real gain. The keyword
    test runs first so that a correct answer which also hedges ("I don't know the
    exact wording, but it is 140 GB") is not miscounted as an abstention. Every
    row carries a blank manual_review column for a human pass.
    """
    answer_norm = _norm(answer)
    keywords = question.get("answer_keywords") or []

    matched, missing = [], []
    for kw in keywords:
        label = " | ".join(kw) if isinstance(kw, (list, tuple)) else str(kw)
        (matched if _keyword_present(answer_norm, kw) else missing).append(label)

    gold_tokens = _tokens(question["gold_answer"])
    answer_tokens = _tokens(answer)
    overlap = (len(gold_tokens & answer_tokens) / len(gold_tokens)) if gold_tokens else 0.0

    abstained = any(p in answer_norm for p in ABSTAIN_PATTERNS)
    if keywords and not missing:
        verdict = "correct"
    elif abstained:
        verdict = "abstained"
    else:
        verdict = "incorrect"

    return {
        "verdict": verdict,
        "matched_keywords": "; ".join(matched),
        "missing_keywords": "; ".join(missing),
        "token_overlap": round(overlap, 3),
    }


# ----------------------------------------------------------------------- the run

def run(args: argparse.Namespace) -> int:
    with open(args.questions, encoding="utf-8") as fh:
        dataset = json.load(fh)

    cache_dir = None if args.no_cache else args.cache_dir

    if args.validate_only:
        return 1 if validate_dataset(dataset, cache_dir) else 0

    require_api_key()

    k_list = sorted({int(k) for k in args.k_list.split(",")})
    max_k = max(k_list)
    video_meta = {v["video_id"]: v for v in dataset["videos"]}

    questions = dataset["questions"]
    if args.videos:
        wanted = set(args.videos.split(","))
        questions = [q for q in questions if q["video_id"] in wanted]
    if args.limit:
        questions = questions[: args.limit]

    by_video: dict[str, list[dict]] = {}
    for q in questions:
        by_video.setdefault(q["video_id"], []).append(q)

    # Embeddings and chat have different free-tier ceilings (100/min vs roughly
    # 10/min), so they get separate budgets. Pacing both off the embedding limit
    # means every answer call bursts past the chat limit and eats a backoff sleep.
    limiter = RateLimiter(args.rpm)
    chat_limiter = RateLimiter(args.chat_rpm)
    rows: list[dict[str, Any]] = []
    index_timings: list[dict[str, Any]] = []

    for vi, (video_id, qs) in enumerate(by_video.items(), start=1):
        meta = video_meta.get(video_id, {})
        print(f"\n[{vi}/{len(by_video)}] {video_id} "
              f"({meta.get('title', 'unknown')}) -- {len(qs)} questions", flush=True)

        text, fetch_s, cached = load_transcript(video_id, cache_dir)

        t0 = time.perf_counter()
        chunks = split_transcript(text)
        split_s = time.perf_counter() - t0

        print(f"  {len(text)} chars -> {len(chunks)} chunks; embedding...", flush=True)
        limiter.take()
        reset_slept()
        t0 = time.perf_counter()
        vector_store = with_quota_retry(
            lambda: build_vector_store(chunks), f"{video_id} embed"
        )
        embed_s = time.perf_counter() - t0 - slept_since_reset()

        retriever_k = build_retriever(vector_store)           # the shipped k=4 retriever
        retriever_max = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": max_k}
        )
        chain = build_chain(retriever_k)

        index_timings.append({
            "video_id": video_id,
            "title": meta.get("title"),
            "duration_min": meta.get("duration_min"),
            "length_class": meta.get("length_class"),
            "transcript_chars": len(text),
            "num_chunks": len(chunks),
            "transcript_from_cache": cached,
            "fetch_s": round(fetch_s, 3),
            "split_s": round(split_s, 4),
            "embed_s": round(embed_s, 3),
            "index_total_s": round(split_s + embed_s, 3),
        })
        print(f"  indexed in {split_s + embed_s:.1f}s "
              f"(embed {embed_s:.1f}s, fetch {fetch_s:.1f}s)", flush=True)

        for qi, q in enumerate(qs, start=1):
            row: dict[str, Any] = {
                "id": q["id"],
                "video_id": video_id,
                "length_class": meta.get("length_class", ""),
                "question": q["question"],
                "gold_answer": q["gold_answer"],
                "gold_substring": q["gold_substring"],
            }

            # One retrieval at max_k. Similarity search returns a ranked list, so
            # the top-4 prefix is exactly what the shipped k=4 retriever returns;
            # --verify-prefix checks that on the first few questions rather than
            # taking it on faith, and saves a query embedding on the rest.
            limiter.take()
            reset_slept()
            t0 = time.perf_counter()
            docs = with_quota_retry(
                lambda: retriever_max.invoke(q["question"]), f"{q['id']} retrieve"
            )
            row["retrieval_s"] = round(time.perf_counter() - t0 - slept_since_reset(), 3)

            needle = _norm(q["gold_substring"])
            normed = [_norm(d.page_content) for d in docs]
            hit_rank = next((i + 1 for i, c in enumerate(normed) if needle in c), -1)
            row["hit_rank"] = hit_rank
            for k in k_list:
                row[f"hit@{k}"] = int(hit_rank != -1 and hit_rank <= k)

            if qi <= args.verify_prefix:
                limiter.take()
                shipped = with_quota_retry(
                    lambda: retriever_k.invoke(q["question"]), f"{q['id']} verify"
                )
                same = [d.page_content for d in shipped] == \
                       [d.page_content for d in docs[:RETRIEVER_K]]
                row["prefix_matches_shipped_retriever"] = int(same)
                if not same:
                    print(f"    [warn] {q['id']}: shipped k={RETRIEVER_K} retriever "
                          f"differs from the top-{RETRIEVER_K} prefix", flush=True)
            else:
                row["prefix_matches_shipped_retriever"] = ""

            if args.no_answers:
                row.update({"answer": "", "verdict": "", "matched_keywords": "",
                            "missing_keywords": "", "token_overlap": "",
                            "answer_s": "", "manual_review": ""})
            else:
                chat_limiter.take()
                reset_slept()
                t0 = time.perf_counter()
                try:
                    answer = with_quota_retry(
                        lambda: answer_question(chain, q["question"]), f"{q['id']} answer"
                    )
                    row["answer_s"] = round(time.perf_counter() - t0 - slept_since_reset(), 3)
                except Exception as exc:  # noqa: BLE001 - record and keep going
                    answer = f"<error: {type(exc).__name__}: {exc}>"
                    row["answer_s"] = ""
                row["answer"] = answer.strip()
                row.update(score_answer(answer, q))
                row["manual_review"] = ""

            rows.append(row)
            flag = "hit " if row.get(f"hit@{RETRIEVER_K}") else "MISS"
            print(f"    {qi:>2}. {flag} rank={hit_rank:<3} "
                  f"{row.get('verdict', ''):<9} {q['id']}", flush=True)

        if vi < len(by_video) and args.sleep_between_videos:
            print(f"  sleeping {args.sleep_between_videos}s before the next video",
                  flush=True)
            time.sleep(args.sleep_between_videos)

    write_outputs(rows, index_timings, dataset, k_list, args)
    return 0


def write_outputs(rows, index_timings, dataset, k_list, args) -> None:
    fields = ["id", "video_id", "length_class", "question", "gold_answer",
              "gold_substring", "hit_rank"] + [f"hit@{k}" for k in k_list] + \
             ["prefix_matches_shipped_retriever", "retrieval_s", "answer",
              "verdict", "matched_keywords", "missing_keywords", "token_overlap",
              "answer_s", "manual_review"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    n = len(rows)
    hit_rates = {}
    for k in k_list:
        hits = sum(r[f"hit@{k}"] for r in rows)
        low, high = _wilson(hits, n)
        hit_rates[f"hit_rate@{k}"] = {
            "hits": hits,
            "n": n,
            "rate": round(hits / n, 4) if n else None,
            "wilson_95ci": [round(low, 4), round(high, 4)],
        }

    answered = [r for r in rows if r.get("verdict")]
    verdicts = {v: sum(1 for r in answered if r["verdict"] == v)
                for v in ("correct", "incorrect", "abstained")}
    n_ans = len(answered)

    groundedness: dict[str, Any] = {"n_answered": n_ans, "counts": verdicts}
    if n_ans:
        for key, count in (("correct", verdicts["correct"]),
                           ("incorrect", verdicts["incorrect"]),
                           ("abstained", verdicts["abstained"])):
            low, high = _wilson(count, n_ans)
            groundedness[f"{key}_rate"] = round(count / n_ans, 4)
            groundedness[f"{key}_wilson_95ci"] = [round(low, 4), round(high, 4)]
        overlaps = [r["token_overlap"] for r in answered
                    if isinstance(r["token_overlap"], (int, float))]
        groundedness["median_token_overlap_vs_gold"] = (
            round(statistics.median(overlaps), 3) if overlaps else None
        )

    per_video = {}
    for timing in index_timings:
        vid = timing["video_id"]
        vrows = [r for r in rows if r["video_id"] == vid]
        entry = {
            "title": timing["title"],
            "duration_min": timing["duration_min"],
            "num_chunks": timing["num_chunks"],
            "index_total_s": timing["index_total_s"],
            "n_questions": len(vrows),
        }
        for k in k_list:
            entry[f"hit_rate@{k}"] = (
                round(sum(r[f"hit@{k}"] for r in vrows) / len(vrows), 3) if vrows else None
            )
        per_video[vid] = entry

    aggregates = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "version": dataset.get("dataset_version"),
            "n_questions": n,
            "n_videos": len({r["video_id"] for r in rows}),
            "annotator": dataset.get("annotator"),
        },
        "pipeline_config_measured": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL,
            "chat_model": CHAT_MODEL,
            "retriever_k_shipped": RETRIEVER_K,
            "search_type": "similarity",
        },
        "retrieval": hit_rates,
        "groundedness": groundedness,
        "latency": {
            "indexing_per_video": _summarise_times(
                [t["index_total_s"] for t in index_timings]
            ),
            "embedding_per_video": _summarise_times(
                [t["embed_s"] for t in index_timings]
            ),
            "retrieval_per_question": _summarise_times(
                [r["retrieval_s"] for r in rows
                 if isinstance(r.get("retrieval_s"), (int, float))]
            ),
            "answer_per_question": _summarise_times(
                [r["answer_s"] for r in rows
                 if isinstance(r.get("answer_s"), (int, float))]
            ),
        },
        "per_video": per_video,
        "index_timings": index_timings,
        "caveats": [
            "Hand-written dataset, single annotator, 30 questions: treat every "
            "rate as +/- its Wilson interval, not as a precise figure.",
            "A hit means a retrieved chunk contains the gold substring. That is a "
            "proxy for relevance, not a human relevance judgement.",
            "Correctness is keyword presence, not an LLM judge. The manual_review "
            "column in results.csv is there for a human pass.",
            "Latency depends on the Gemini API and the network on the day; it is "
            "not a benchmark of the local code.",
        ],
    }

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(aggregates, fh, indent=2)

    print("\n" + "=" * 62)
    print(f"{n} questions over {aggregates['dataset']['n_videos']} videos")
    for k in k_list:
        hr = hit_rates[f"hit_rate@{k}"]
        print(f"  hit rate @k={k:<2} {hr['rate']:.3f}  ({hr['hits']}/{hr['n']}, "
              f"95% CI {hr['wilson_95ci'][0]:.2f}-{hr['wilson_95ci'][1]:.2f})")
    if n_ans:
        print(f"  correct    {verdicts['correct']}/{n_ans} "
              f"({groundedness['correct_rate']:.3f})")
        print(f"  incorrect  {verdicts['incorrect']}/{n_ans} "
              f"({groundedness['incorrect_rate']:.3f})")
        print(f"  abstained  {verdicts['abstained']}/{n_ans} "
              f"({groundedness['abstained_rate']:.3f})")
    lat = aggregates["latency"]
    for label, key in (("index/video ", "indexing_per_video"),
                       ("retrieval/q ", "retrieval_per_question"),
                       ("answer/q    ", "answer_per_question")):
        stats = lat[key]
        if stats["n"]:
            print(f"  {label} median {stats['median_s']:.2f}s  p90 {stats['p90_s']:.2f}s")
    print("=" * 62)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_json}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=os.path.join(_HERE, "questions.json"))
    parser.add_argument("--out-csv", default=os.path.join(_HERE, "results.csv"))
    parser.add_argument("--out-json", default=os.path.join(_HERE, "results.json"))
    parser.add_argument("--cache-dir", default=os.path.join(_HERE, "cache"),
                        help="where fetched transcripts are cached")
    parser.add_argument("--no-cache", action="store_true",
                        help="always refetch transcripts")
    parser.add_argument("--validate-only", action="store_true",
                        help="check gold substrings against the chunks, no API calls")
    parser.add_argument("--no-answers", action="store_true",
                        help="retrieval + latency only, skip the chain")
    parser.add_argument("--k-list", default=",".join(str(k) for k in DEFAULT_K_LIST))
    parser.add_argument("--videos", default="", help="comma-separated video ids to limit to")
    parser.add_argument("--limit", type=int, default=0, help="first N questions only")
    parser.add_argument("--rpm", type=int, default=45,
                        help="embedding/retrieval operations per minute ceiling (free "
                             "tier allows 100 embedding requests/min)")
    parser.add_argument("--chat-rpm", type=int, default=8,
                        help="chat-model calls per minute ceiling. The free tier allows "
                             "far fewer of these than embeddings (roughly 10/min), so "
                             "this is paced separately; raise it on a paid key")
    parser.add_argument("--verify-prefix", type=int, default=3,
                        help="per video, confirm the shipped k=%d retriever matches the "
                             "top-%d prefix for this many questions" % (RETRIEVER_K, RETRIEVER_K))
    parser.add_argument("--sleep-between-videos", type=float, default=5.0)
    args = parser.parse_args()

    try:
        return run(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
