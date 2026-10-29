# RAGAS baseline (lần chạy lại 2026-10-29, chưa chú giải)

Judge/LLM: `kr/claude-sonnet-5.5` · RAGAS-evaluated questions: 25

RAGAS version: 0.4.3 · API: legacy ragas.metrics classes (deprecated shims) + LangchainLLMWrapper/LangchainEmbeddingsWrapper + evaluate

Runtime: 953s

## Overall means

| metric | mean | NaN count |
|---|---|---|
| faithfulness | 0.961 | 0 |
| llm_context_precision_with_reference | 0.309 | 0 |
| context_recall | 0.420 | 0 |
| answer_relevancy | 0.665 | 0 |

## By category

| category | faithfulness | llm_context_precision_with_reference | context_recall | answer_relevancy |
|---|---|---|---|---|
| combined | 0.966 | 0.000 | 0.433 | 0.585 |
| numeric | 0.967 | 0.339 | 0.600 | 0.532 |
| risk | 0.954 | 0.434 | 0.233 | 0.837 |

## Intent accuracy (all 30 questions, incl. traps): 30/30

## Numeric sanity (key figure appears in answer): 6/10

## Caveats

- The judge model (`kr/claude-sonnet-5.5`) is the same model as the application LLM, so it grades its own output; this likely inflates faithfulness.

- 0 app failures in this run (a failed row would be scored 0.0 and listed in `meta.errors`).

- answer_relevancy is 0.0 for refusal or noncommittal answers, even when the answer is honest or grounded; this is a RAGAS design artifact, not necessarily a bad answer.

- Numeric sanity is a substring check: a hit can be a derived figure (for example a subtraction) rather than one stated directly, and substring matching of short figures is loose.

- A number can be grounded but misread: `ground_numbers` and faithfulness only check that the figure appears in (or is supported by) a retrieved chunk, not what it means, so a figure taken from a table with missing headers can pass both checks while answering a different question.

- The retrieved contexts passed to the judge are not stored in the JSON, so judge inputs cannot be audited after the fact.

## Worst cases (for Debug questions)

- **q18** (faithfulness 0.727): What manufacturing and supply dependency risks does NVIDIA disclose?
  - Why: TO ANALYZE
- **q05** (faithfulness 0.800): What was Microsoft's net income in its most recent fiscal year?
  - Why: TO ANALYZE
- **q14** (faithfulness 0.900): What risks related to artificial intelligence does Microsoft disclose?
  - Why: TO ANALYZE
- **q21** (faithfulness 0.900): What is Apple's current stock price, and what were its total net sales last fiscal year?
  - Why: TO ANALYZE
- **q24** (faithfulness 0.929): Compare Apple's and Microsoft's latest annual revenue and tell me both current stock prices.
  - Why: TO ANALYZE
