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

## Worst cases (for Debug questions)

- **q05** (faithfulness 0.714): What was Microsoft's net income in its most recent fiscal year?
  - Why: Retrieval miss. The golden chunk MSFT-2026-8-001 (income statement) was not among the 5 retrieved (MSFT-2026-8-079, 7-002, 8-134, 7-018, 7-020; best distance 0.439). The answer correctly declined ("context doesn't include Microsoft's net income for fiscal year 2026") and cited operating income $155,237M from 7-020 instead; the judge scored this 0.71 faithfulness, and recall/precision/relevancy were all 0.0. The answer is honest, not a hallucination; the failure is retrieval of the statements chunk.
- **q04** (faithfulness 0.714): What was Microsoft's total revenue in its most recent fiscal year?
  - Why: Right answer from a different chunk. The golden chunk MSFT-2026-8-000 was not retrieved; the answer ($331,839M, +18% vs $281,724M) cites MSFT-2026-7-020 (MD&A table). Faithfulness 0.71 likely because the answer adds a hedge about the table's missing column headers ("I'm reading the first column as FY2026"), an inference the context does not state outright. Recall is 1.0 (the figure is in the context) while precision is 0.25 (golden chunk absent, 4 of 5 chunks off-target).
- **q21** (faithfulness 0.818): What is Apple's current stock price, and what were its total net sales last fiscal year?
  - Why: Retrieval miss on the filing half. The golden chunk AAPL-2025-7-005 was not retrieved; all 5 chunks were Item 8 (8-077, 8-041, 8-006, 8-003, 8-004), so the FY2025 net sales total never appeared. The answer hedged correctly ("context doesn't include Apple's total net sales for fiscal 2025"), gave the price from the tool, and quoted only FY2023 total and U.S. sales. Faithfulness 0.82 is dragged down by the tool price/market-cap statements and the tangential figures; recall 0.5 reflects the price half being covered while the sales half is missing.
- **q24** (faithfulness 0.867): Compare Apple's and Microsoft's latest annual revenue and tell me both current stock prices.
  - Why: Retrieval miss on both companies. Neither golden chunk (AAPL-2025-7-005, MSFT-2026-8-000) was in the 10 retrieved chunks (distances 0.53 to 0.58, the weakest of the set). The answer refused the like-for-like comparison, citing only FY2023 Apple total, pro forma Microsoft figures and Microsoft Cloud revenue. It hedged rather than invented; recall and precision are 0.0.
- **q03** (faithfulness 0.889): How much net sales did Apple's iPhone segment generate in its most recent fiscal year?
  - Why: Retrieval near-miss. The golden chunk AAPL-2025-7-006 was not retrieved (AAPL-2025-7-005 and 8-023 were). The answer noted the iPhone row is cut off at the start of the excerpt in 8-023 and began inferring iPhone as total minus the other categories (the answer ends with an "If those are the only categories" derivation). That derived figure is not stated in any chunk, which is the likely source of the 0.89 faithfulness. Recall is 1.0 but precision 0.0.

## Common pattern

In 5 of 5 worst cases the golden chunk id cited in the ground truth was not retrieved; the model hedged honestly rather than fabricating. The NaN count is 0 for all metrics. Numeric sanity misses (q05, q06, q10) are retrieval misses of the same kind. Scores are unedited.
