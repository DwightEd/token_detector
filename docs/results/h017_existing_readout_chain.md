# M4 existing readout-chain audit:15716

**The old unit warning was already present at "ago"; later R4/V3 readout lowered its global risk rank.** This is a descriptive account within the OLD system, not a cause inferred from M4.

Case15716/source15540; original generator `mistral-7B-instruct`, historical observer `Meta-Llama-3.1-8B-Instruct`. Gap=log p(absent source) - log p(present source), with original response history. Coordinates are half-open. The original QA budget was6241/124817 tokens; all ranks below are reused, not recomputed.

|Target|Characters|Text|Old source p/full|Old absent p/full|Full/local gap|R4/V3 final ranks|
|---|---|---|---|---|---|---|
|56|[266,273)|` around`|0.068413365|0.064842992|-0.053599/+2.314138|115312/77574|
|57|[273,274)|` `|0.99588261|0.91355111|-0.086290/-0.015751|117909/75223|
|58|[274,277)|`300`|0.99798502|0.011336294|-4.477729/-4.969757|117102/78404|
|59|[277,283)|` years`|0.92260135|0.005819485|-5.065985/-6.257875|116900/63082|
|60|[283,287)|` ago`|0.000121027|0.0032453017|+3.288950/+2.862774|114453/20219|

"300" and "years" are source-predictable components; the OLD "ago" already has low probability and a strong positive source-deletion gap. This does not support high-probability wrong temporal binding for the whole expression. These are observer probabilities, not original-generator probabilities or factuality probabilities.

|Stored "ago" stage|Value|Old QA risk rank|
|---|---|---|
|full gap|+3.288949966|1165|
|local gap|+2.862773895|3975|
|R4 actual pair point|+3.075861931|1492|
|R4 shared unit prior|-1.67420571|116867|
|R4 final|-1.111751692|114453|
|V3 own witness emission|+2.632417209|1165|
|V3 final|+0.3493939088|20219|

The first R4 loss is at the **shared sentence prior, before route combination**: actual point rank1492 -> prior116867. The22-token unit spans characters[178,288), the entire sentence beginning "The first known reference". Its stored shared prior is -1.6742057096;19/22 existing pair contrasts are negative. The frozen QA rule broadcasts log(mean(exp(0.1*point)))/0.1 over that unit. Thus the isolated unit warning is combined with many source-predictable words. This is not evidence that sharing is harmful overall.

V3 stores own emission **2.6324172092**, context difference **-2.2830233004**, final **0.3493939088**, rank **20219**. The warning remains positive but falls outside the budget. The context difference is an algebraic decomposition, not a deletion intervention. No parameters or scoring rules were changed.

**Cross-run causal bridge stopped.** All365 prompt IDs,76 answer IDs, source-mask positions and answer offsets match M4 exactly, notwithstanding the `source_first_v1` directory name. But the predeclared absolute logp budget0.002 fails **44/76** values; max difference **0.12790775299**. At "ago", old logp -9.0194969177 vs M4 -9.0729827881 differs by0.05348587036. Old computation declares BF16; M4 uses BF16 stored weights/FP32 computation. The old weight/code identity is not authenticated by metadata equality. No tolerance widening or joint causal attribution is permitted.

The separate sealed M4 observation is compatible with attending to BC and assigning low p to "ago"; that is not merged into the old score computation. Source deletion also recomputes positions, so even the old gap is a predictive contrast rather than isolated factual causality.

**Strict arithmetic detail retained:** native delta and witness full_gap agree bitwise. Primitive subtraction versus native delta differs in two full-channel cells by at most4.440892098500626e-16 (one FP64 ULP here). Exact primitive arithmetic identity did not pass these two cells; no stored values were changed, and the historical run is not declared failed. Its impact on the stated large gap/rank result is negligible. Exact signed differences are preserved in JSON.

23 input hashes bind the old manifest, primitive source/response/probability records, native/witness plans/seals/632.npz, published case diagnostics and M4 endpoints. JSON preserves all76 cross-run differences, six neighboring tokens, historical model/config/code/protocol/reference contracts, and raw strict-check differences. Global ranks come from the already-published diagnostics. No annotations/evaluation-only files, new target labels, weights, models, scores, full-score recomputation, ranking recomputation or downloads were used.

Native seal `7537baa6d28ca723e6e1697f20cbfccde04a8c72e126b2ddc24324bf6be67d48`; witness seal `7a0034462d35588c0b7883802288b074727ed2bc28bf3a0a78c887c9ebada88d`.

JSON SHA256: `c05194562dcb75c1ee8c0265521e6cf4ac1b2f733912093537726fa0fd7b698c`.
