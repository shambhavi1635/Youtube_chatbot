# Retrieval evaluation

A small measured evaluation of the RAG pipeline in `backend/pipeline.py`, so the
project can quote results rather than configuration values.

The harness imports `split_transcript`, `build_vector_store`, `build_retriever`,
`build_chain` and `answer_question` from the pipeline and changes no tuning
constant. It measures the system as it ships: `chunk_size=2000`,
`chunk_overlap=100`, `gemini-embedding-001` embeddings, FAISS similarity search,
`k=4`.

## Dataset

`questions.json` — 30 questions over 5 videos, 6 per video, written by hand
against the fetched transcripts.

| Video | Length | Chunks | Questions |
|---|---|---|---|
| `MMmOLN5zBLY` The benefits of a bilingual brain (TED-Ed) | 4.8 min | 3 | 6 |
| `aircAruvnKk` But what is a neural network? (3Blue1Brown) | 18.4 min | 11 | 6 |
| `iG9CE55wbtY` Do schools kill creativity? (Ken Robinson, TED) | 19.4 min | 10 | 6 |
| `wjZofJX0v4M` But what is a GPT? (3Blue1Brown) | 26.9 min | 16 | 6 |
| `zjkBMFhNj_g` Intro to Large Language Models (Karpathy) | 59.8 min | 34 | 6 |

Each question carries:

- `question` — the question as a user would type it
- `gold_answer` — a short reference answer
- `gold_substring` — an exact phrase from the transcript passage that answers it
- `answer_keywords` — the facts an answer must contain to count as correct

Transcripts are fetched with the repo's own `fetch_transcript()`, so the indexed
text is exactly what the app would index, and cached under `eval/cache/` so
repeat runs stay comparable if captions change upstream.

Every gold substring is verified to fall inside a single chunk before any
scoring runs (`--validate-only`). A phrase that straddled a chunk boundary could
never be retrieved at any `k`, and that would be a dataset bug rather than a
retrieval result. All 30 pass, and the answer chunks are spread across each
video rather than clustered at the start — including chunk 32 of 34 in the
hour-long video.

## Metrics

**Retrieval hit rate at k = 1, 4, 8.** A question is a hit at `k` if any of the
top-`k` retrieved chunks contains its gold substring. Reported as a curve so the
`k=4` figure can be read against what more or less retrieval would buy.
Matching is whitespace-normalised and case-insensitive (captions break lines
mid-sentence) and digit group separators are stripped, so `13,000` matches
`13000`.

Retrieval runs once per question at the largest `k`; because similarity search
returns a ranked list, the top-4 prefix is what the shipped `k=4` retriever
returns. The harness does not take that on faith — for the first few questions
of each video it also calls the shipped retriever and asserts the lists match
(`prefix_matches_shipped_retriever` in the CSV).

**Groundedness.** The full chain runs on each question and the answer is
classified `correct` / `incorrect` / `abstained`. Correctness is keyword
presence; there is no LLM judge, which for 30 questions would add cost and a
second source of error. The keyword test runs *before* the abstention test, so a
correct answer that also hedges is not miscounted as an abstention.

Abstentions are reported separately, not folded into `incorrect`: the prompt in
`pipeline.py` explicitly instructs the model to say it does not know when the
context is insufficient, so an abstention is the prompt working as designed, and
is a different failure from a confident wrong answer.

**Latency.** Indexing time per video and query time per question, each as median
and p90.

All rates come with a 95% Wilson score interval, because on 30 questions the
interval is wide and quoting a bare percentage would overstate the precision.

## Running it

```bash
python eval/retrieval_eval.py --validate-only   # no API key, checks the dataset
python eval/retrieval_eval.py --no-answers      # retrieval + latency only
python eval/retrieval_eval.py                   # everything
```

Needs `GOOGLE_API_KEY` in `.env` for anything but `--validate-only`. The script
loads the repo's `.env` by absolute path and resolves its own inputs and outputs
relative to itself, so it can be run from any working directory.

Embeddings and the chat model have different free-tier ceilings, so they are
paced separately: `--rpm` (default 45) covers embedding and retrieval calls
against a 100/min limit, and `--chat-rpm` (default 8) covers answer generation,
where the free tier allows roughly 10/min. Pacing both off the embedding limit
makes every answer call overrun the chat limit and eat a backoff sleep. On a paid
key, raise `--chat-rpm`.

Quota errors (`429` / `RESOURCE_EXHAUSTED`, which `pipeline._is_quota_error`
already detects) are retried with increasing backoff rather than aborting a run
partway. A full run is ~74 document embeddings plus about 2 calls per question,
and the chat pacing makes it roughly 4-5 minutes on a free key.

Running the Streamlit app at the same time competes for the same quota.

Outputs: `results.csv` (per question) and `results.json` (aggregates, per-video
breakdown, timings, caveats).

## Results

<!-- Populated by a live run; see results.json for the full aggregates. -->

Not yet populated — needs a `GOOGLE_API_KEY`. Run the command above and the
numbers land in `results.csv` / `results.json`.

## Limitations

These are real but small numbers, and worth stating plainly:

- **30 questions is a small dataset.** The 95% Wilson interval on a 30-item rate
  is roughly ±10–15 points. Treat every figure as a range, not a point.
- **Single annotator.** One person wrote the questions, the gold answers and the
  keywords, having read the transcripts. There is no second annotator and no
  inter-annotator agreement, so question difficulty and phrasing carry that
  person's bias — including a bias toward questions with crisp factual answers,
  which are easier to retrieve than vague or synthesising ones.
- **Substring matching is a proxy for relevance.** A chunk containing the gold
  phrase is assumed relevant and one without it is assumed not. A chunk that
  answers the question in different words counts as a miss, so the hit rate is a
  slight under-estimate of true retrieval quality.
- **Keyword scoring is a proxy for correctness.** It cannot see a fluent answer
  that contains the right keyword in a wrong claim. `results.csv` carries a blank
  `manual_review` column for a human pass over the 30 answers.
- **Questions are answerable by construction.** Each was written from a passage
  that exists in the transcript, so this measures retrieval over in-scope
  questions. It says nothing about behaviour on out-of-scope questions, where the
  abstention rate is the number that would matter.
- **Five videos, mostly explainer content.** Four are technical or educational
  talks with clean manual captions. Auto-generated captions, accented speech,
  music or multi-speaker audio are not represented.
- **Latency is not a benchmark of this code.** It is dominated by Gemini API
  round trips and the network on the day of the run.
