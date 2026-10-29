# RAGAS baseline

Judge/LLM: `kr/claude-sonnet-5.5` · RAGAS-evaluated questions: 25

RAGAS version: 0.4.3 · API: legacy ragas.metrics classes (deprecated shims) + LangchainLLMWrapper/LangchainEmbeddingsWrapper + evaluate

Runtime: 814s

## Overall means

| metric | mean | NaN count |
|---|---|---|
| faithfulness | 0.939 | 0 |
| llm_context_precision_with_reference | 0.309 | 0 |
| context_recall | 0.387 | 0 |
| answer_relevancy | 0.687 | 0 |

## By category

| category | faithfulness | llm_context_precision_with_reference | context_recall | answer_relevancy |
|---|---|---|---|---|
| combined | 0.903 | 0.000 | 0.333 | 0.561 |
| numeric | 0.926 | 0.339 | 0.600 | 0.615 |
| risk | 0.971 | 0.434 | 0.200 | 0.822 |

## Intent accuracy (all 30 questions, incl. traps): 30/30

## Numeric sanity (key figure appears in answer): 7/10

## Caveats

- The judge model (`kr/claude-sonnet-5.5`) is the same model as the application LLM, so it grades its own output; this likely inflates faithfulness.

- 0 app failures in this run (a failed row would be scored 0.0 and listed in `meta.errors`).

- answer_relevancy is 0.0 for refusal or noncommittal answers, even when the answer is honest or grounded; this is a RAGAS design artifact, not necessarily a bad answer. It affects 6 rows: q05, q06, q10, q18, q21, q24 (q21 gave a real price from the tool yet scored 0.0).

- Numeric sanity is a substring check: a hit can be a derived figure (for example a subtraction) rather than one stated directly, and substring matching of short figures is loose. The q03 hit (209,586) is such a derived figure: the answer computed it by subtraction and said so, it is not stated in the filing excerpt.

- A number can be grounded but misread: `ground_numbers` and faithfulness only check that the figure appears in (or is supported by) a retrieved chunk, not what it means, so a figure taken from a table with missing headers can pass both checks while answering a different question.

- Confirmed instance (hand-verified, q10 and q20): both answers present "total revenue $139,297" (vs $87,960 prior year), taken from chunk NVDA-2026-7-029, a headerless segment table whose rows are Compute & Networking 130,141, Graphics 9,156 and Total 139,297 (130,141 + 9,156 = 139,297; it sits before the "Reportable segment operating income" text and its prior-year total 87,960 does not match revenue of $130,497M). NVIDIA's true FY2026 total revenue is $215,938M (q07 answers it correctly). The structured `revenue` field was 139297 in both rows and passed `ground_numbers` because that string is in a retrieved chunk. The judge still scored q20 faithfulness 1.0 and q10 0.941, since the figure is literally in the context. The q10 answer at least flags the truncated label and missing units; the q20 answer says only that units are not shown. So q10 and q20 are not pure retrieval hedges: q20 (and the `revenue` field of q10) contain a wrong number that the automated checks did not catch.

- The retrieved contexts passed to the judge are not stored in the JSON, so judge inputs cannot be audited after the fact.

## Worst cases (for Debug questions)

- **q05** (faithfulness 0.714): What was Microsoft's net income in its most recent fiscal year?
  - Why: Retrieval miss. The golden chunk MSFT-2026-8-001 (income statement) was not among the 5 retrieved (MSFT-2026-8-079, 7-002, 8-134, 7-018, 7-020; best distance 0.439). The answer correctly declined ("context doesn't include Microsoft's net income for fiscal year 2026") and cited operating income $155,237M from 7-020 instead; the judge scored this 0.71 faithfulness, and recall/precision/relevancy were all 0.0. The answer is honest, not a hallucination; the failure is retrieval of the statements chunk.
- **q04** (faithfulness 0.714): What was Microsoft's total revenue in its most recent fiscal year?
  - Why: Right answer from a different chunk. The golden chunk MSFT-2026-8-000 was not retrieved; the answer ($331,839M, +18% vs $281,724M) cites MSFT-2026-7-020 (MD&A table). Faithfulness 0.71 likely because the answer adds a hedge about the table's missing column headers ("I'm reading the first column as FY2026"), an inference the context does not state outright. Recall is 1.0 (the figure is in the context) while precision is 0.25 (golden chunk absent, 4 of 5 chunks off-target).
- **q21** (faithfulness 0.818): What is Apple's current stock price, and what were its total net sales last fiscal year?
  - Why: Retrieval miss on the filing half. The golden chunk AAPL-2025-7-005 was not retrieved; all 5 chunks were Item 8 (8-077, 8-041, 8-006, 8-003, 8-004), so the FY2025 net sales total never appeared. The answer hedged correctly ("context doesn't include Apple's total net sales for fiscal 2025"), gave the price from the tool, and quoted only FY2023 total and U.S. sales. Faithfulness 0.82 is dragged down by the tool price/market-cap statements and the tangential figures; recall 0.5 reflects the price half being covered while the sales half is missing.
- **q24** (faithfulness 0.867): Compare Apple's and Microsoft's latest annual revenue and tell me both current stock prices.
  - Why: Retrieval miss on both companies. Neither golden chunk (AAPL-2025-7-005, MSFT-2026-8-000) was in the 10 retrieved chunks (distances 0.53 to 0.58, among the weakest). The answer refused the like-for-like comparison, citing only FY2023 Apple total, pro forma Microsoft figures and Microsoft Cloud revenue. It hedged rather than invented; recall and precision are 0.0.
- **q03** (faithfulness 0.889): How much net sales did Apple's iPhone segment generate in its most recent fiscal year?
  - Why: Retrieval near-miss. The golden chunk AAPL-2025-7-006 was not retrieved (AAPL-2025-7-005 and 8-023 were). The answer noted the iPhone row is cut off at the start of the excerpt in 8-023 and began inferring iPhone as total minus the other categories (the answer ends with an "If those are the only categories" derivation). That derived figure is not stated in any chunk, which is the likely source of the 0.89 faithfulness. Recall is 1.0 but precision 0.0.

## Common pattern

In 5 of 5 worst cases the golden chunk id cited in the ground truth was not retrieved; in those five cases the model hedged or answered from another chunk instead of inventing a figure. The NaN count is 0 for all metrics. Numeric sanity misses (q05, q06, q10) all start as retrieval misses, but they did not all end as hedges: in q10 (and q20) the answer used a misread, headerless table total (see Caveats), so the system does not always hedge when retrieval fails. Scores are unedited.
