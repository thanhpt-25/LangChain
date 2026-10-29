# RAGAS both_rewrite_1

Judge/LLM: `kr/claude-sonnet-5.5` · RAGAS-evaluated questions: 25

RAGAS version: 0.4.3 · API: legacy ragas.metrics classes (deprecated shims) + LangchainLLMWrapper/LangchainEmbeddingsWrapper + evaluate

Runtime: 836s

## Overall means

| metric | mean | NaN count |
|---|---|---|
| faithfulness | 0.952 | 0 |
| llm_context_precision_with_reference | 0.377 | 0 |
| context_recall | 0.420 | 0 |
| answer_relevancy | 0.772 | 0 |

## By category

| category | faithfulness | llm_context_precision_with_reference | context_recall | answer_relevancy |
|---|---|---|---|---|
| combined | 0.983 | 0.282 | 0.333 | 0.769 |
| numeric | 0.909 | 0.409 | 0.600 | 0.609 |
| risk | 0.980 | 0.392 | 0.283 | 0.937 |

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

- **q02** (faithfulness 0.750): What was Apple's diluted earnings per share in its most recent fiscal year?
  - Why: Retrieval is fine: AAPL-2025-8-026 (holds 7.46 and the diluted label) is the top-1 chunk and the answer states $7.46 correctly, with context recall 1.0 and precision 1.0. Faithfulness 0.750 comes from the closing sentence that says the EPS table has no column headers and the first value was read as 2025; that statement is about the context, not a claim the judge could match to a chunk. Same question scored 1.0 and 0.88 in earlier runs and 1.0 in the second new run with the same chunks, so this is judge variance.
- **q03** (faithfulness 0.750): How much net sales did Apple's iPhone segment generate in its most recent fiscal year?
  - Why: Retrieval is fine: AAPL-2025-7-006 (holds 209,586 for iPhone) is in the top-5 and the answer gives $209,586 million with the Pro-models explanation from 7-007. The answer also adds that iPhone is a category, not a reportable segment, and that a later fiscal year may have ended; the judge did not find these in a chunk. Faithfulness 0.75 equals demo_check and the chunks are identical in all four runs, so this is a stable property of the answer wording, not of the rewrite. Precision 0.25 is low because only 1 of 5 chunks is relevant.
- **q04** (faithfulness 0.846): What was Microsoft's total revenue in its most recent fiscal year?
  - Why: Retrieval miss, unchanged: the chunk with 331,839 and Total revenue is not in the top-5 (MSFT-2026-8-134, 7-002, 7-017, 8-079, 7-021, the same five as every earlier run). The answer says total revenue is not in the context and gives only the 18% growth ($50.1 billion) and Microsoft Cloud $214.4 billion, then correctly flags the pro forma $247,442 million as an older year. Recall, precision and relevancy are 0.0. Faithfulness 0.846 reflects a partly supported side statement. This is a numeric question, so the rewrite does not apply.
- **q09** (faithfulness 0.875): What was NVIDIA's diluted earnings per share in its most recent fiscal year?
  - Why: Retrieval is fine: NVDA-2026-15-020 is in the top-5 and the answer states $4.90 versus $2.94 and $1.19, with recall 1.0 and precision 1.0. Faithfulness 0.875 comes from the last sentence saying the EPS rows have no column headers and the years were matched by the order of adjacent statements, which is an inference about the context. The same 0.86 to 0.88 appears in all four runs.
- **q11** (faithfulness 0.882): What supply chain risks does Apple disclose in its 10-K?
  - Why: Retrieval miss, unchanged: the two evidence groups (single-source partners, tariffs and import controls) are not among AAPL-2025-1A-031, -021, -056, -003, -055, the same top-5 as all earlier runs. The answer lists supply shortages, supplier financial stress and supplier cybersecurity from those chunks and omits the single-source and tariff points, so recall and precision are 0.0. Faithfulness 0.882 comes from one or two bullets the judge found only partly supported. Not related to the rewrite (filing question, original query used).
