# RAGAS baseline

Judge/LLM: `kr/claude-sonnet-5.5` · RAGAS-evaluated questions: 25

RAGAS version: 0.4.3 · API: legacy ragas.metrics classes (deprecated shims) + LangchainLLMWrapper/LangchainEmbeddingsWrapper + evaluate

Runtime: 1134s

## Overall means

| metric | mean | NaN count |
|---|---|---|
| faithfulness | 0.939 | 0 |
| llm_context_precision_with_reference | 0.342 | 0 |
| context_recall | 0.400 | 0 |
| answer_relevancy | 0.766 | 0 |

## By category

| category | faithfulness | llm_context_precision_with_reference | context_recall | answer_relevancy |
|---|---|---|---|---|
| combined | 0.922 | 0.107 | 0.267 | 0.768 |
| numeric | 0.911 | 0.409 | 0.600 | 0.609 |
| risk | 0.977 | 0.392 | 0.267 | 0.921 |

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

- **q07** (faithfulness 0.600): What was NVIDIA's total revenue in its most recent fiscal year?
  - Why: Retrieval was not the problem: top-5 held NVDA-2026-15-018 (the benchmark counts q07 as a hit) and the answer states 215,938 correctly. The low faithfulness (0.600, was 1.0 in the dense run with similar contexts) comes from the closing sentence saying the chunk cuts off after the first value and that prior-year figures are unavailable, a statement about the context that the judge could not match to a claim in it; the stored data does not show the judge's reasoning. Context recall is still 0.0 because the golden chunk NVDA-2026-7-012 (MD&A, with the 65% growth and prior year) was not retrieved.
- **q21** (faithfulness 0.778): What is Apple's current stock price, and what were its total net sales last fiscal year?
  - Why: Evidence is in the top-5 (AAPL-2025-8-023 is cited and the benchmark counts q21 as a hit), the answer gives 416,161 and the live price, and context recall is 0.5 (the price half comes from the tool, not a chunk). Faithfulness 0.778 is below both earlier runs (0.82 and 0.90): the answer adds an inference that FY2026 has likely just ended, which no retrieved chunk supports. This question improved on retrieval but not on the judged scores beyond precision (0.20 vs 0.0).
- **q11** (faithfulness 0.850): What supply chain risks does Apple disclose in its 10-K?
  - Why: No retrieval fix: the two evidence groups ('relies on single-source partners', 'tariffs and other controls on imports') are not in the top-5 and were outside the 30 candidates in the benchmark, so the reranker could not promote them. The top-5 (AAPL-2025-1A-031, -021, -056, -003, -055) are generic supply and security risks; the answer lists those and omits single-source and tariffs. Faithfulness 0.85 reflects one or more answer statements the judge found only partly supported; context precision and recall are 0.0.
- **q05** (faithfulness 0.867): What was Microsoft's net income in its most recent fiscal year?
  - Why: Still a retrieval miss: the net income chunk (the chunk holding '133,749 ... Net income', evidence rank 12 in the benchmark) is not in the top-5, which holds MSFT-2026-8-079 (a pro forma table), 8-134, 7-017, 7-002 and 8-136. The model hedged instead of inventing a figure, flagged the $88,308 million pro forma number as not the reported net income, and gave no answer, so recall, precision and relevancy are 0.0. Same outcome as both earlier runs; faithfulness 0.867 is between them (0.714 and 0.800).
- **q04** (faithfulness 0.875): What was Microsoft's total revenue in its most recent fiscal year?
  - Why: Regression versus both earlier runs. The evidence chunk (the chunk holding '331,839 ... Total revenue', benchmark rank 7) is not in the top-5, so the answer says total revenue is not in the context and gives only the 18% growth and Microsoft Cloud figures. In the dense run the same question had recall 1.0 and a correct answer (relevancy 0.88). The dense top-5 of the earlier run contained MSFT-2026-7-020, which is absent now (the reranker replaced it with 7-017 and 8-136); the benchmark puts the evidence at rank 7 for this configuration, so precision, recall and relevancy fall to 0.0.
