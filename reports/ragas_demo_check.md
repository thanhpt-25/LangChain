# RAGAS demo_check (lần chạy thứ hai của cấu hình cuối: dense + reranker MiniLM, 2026-10-29)

Judge/LLM: `kr/claude-sonnet-5.5` · RAGAS-evaluated questions: 25

RAGAS version: 0.4.3 · API: legacy ragas.metrics classes (deprecated shims) + LangchainLLMWrapper/LangchainEmbeddingsWrapper + evaluate

Runtime: 971s

## Overall means

| metric | mean | NaN count |
|---|---|---|
| faithfulness | 0.933 | 0 |
| llm_context_precision_with_reference | 0.342 | 0 |
| context_recall | 0.413 | 0 |
| answer_relevancy | 0.770 | 0 |

## By category

| category | faithfulness | llm_context_precision_with_reference | context_recall | answer_relevancy |
|---|---|---|---|---|
| combined | 0.978 | 0.107 | 0.200 | 0.772 |
| numeric | 0.878 | 0.409 | 0.650 | 0.603 |
| risk | 0.966 | 0.392 | 0.283 | 0.937 |

## Intent accuracy (all 30 questions, incl. traps): 30/30

## Numeric sanity (key figure appears in answer): 7/10

## Caveats

- The judge model (`kr/claude-sonnet-5.5`) is the same model as the application LLM, so it grades its own output; this likely inflates faithfulness.

- 0 app failures in this run (a failed row would be scored 0.0 and listed in `meta.errors`).

- answer_relevancy is 0.0 for refusal or noncommittal answers, even when the answer is honest or grounded; this is a RAGAS design artifact, not necessarily a bad answer.

- Numeric sanity is a substring check: a hit can be a derived figure (for example a subtraction) rather than one stated directly, and substring matching of short figures is loose.

- A number can be grounded but misread: `ground_numbers` and faithfulness only check that the figure appears in (or is supported by) a retrieved chunk, not what it means, so a figure taken from a table with missing headers can pass both checks while answering a different question.

- The retrieved contexts passed to the judge are not stored in the JSON, so judge inputs cannot be audited after the fact.

## Worst cases (for Debug questions)

- **q01** (faithfulness 0.714): What were Apple's total net sales in its most recent fiscal year?
  - Why: TO ANALYZE
- **q03** (faithfulness 0.750): How much net sales did Apple's iPhone segment generate in its most recent fiscal year?
  - Why: TO ANALYZE
- **q04** (faithfulness 0.846): What was Microsoft's total revenue in its most recent fiscal year?
  - Why: TO ANALYZE
- **q10** (faithfulness 0.846): How much revenue did NVIDIA's Data Center market platform generate in its most recent fiscal year?
  - Why: TO ANALYZE
- **q11** (faithfulness 0.846): What supply chain risks does Apple disclose in its 10-K?
  - Why: TO ANALYZE
