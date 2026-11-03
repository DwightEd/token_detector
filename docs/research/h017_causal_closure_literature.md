# M4 causal-closure literature audit

Audit date: 2026-11-03 (Asia/Shanghai). Primary sources only. This is a read-only literature audit, not scientific execution or algorithm admission. No weights were downloaded or loaded; no model calls, target labels, source-code edits, or persistent research-state changes were made.

## Finding

The architectural premise is established prior work: a causal decoder's state at an early token cannot depend on later realized input tokens. Echo embeddings, LLM2Vec, re-reading, and recent sequence-labeling work already address this limitation. There is also a direct hallucination-localization precedent using answer repetition. Neither “later states see a completed clause” nor “repeat text to obtain right-context-aware token features” is a defensible novelty claim by itself.

The reviewed papers do not establish the particular diagnosis that the frozen natural response R15716 was caused by using the numeral's incomplete source K/V instead of a later era-complete representation. That remains unestablished, rather than demonstrated novel. The proposed unchanged single-case M4 observation cannot establish causality or population detection gains.

## Exact architectural statement and its limits

Let position j be the source numeral and let k>j contain the disambiguating suffix, under a fixed tokenizer, positions and strictly causal decoder mask. With dropout disabled, every layer satisfies

\[
h_j^{(l)}=F_j^{(l)}(x_{\leq j}),\qquad
K_j^{(l)}=R_jW_K^{(l)}h_j^{(l-1)},\qquad
V_j^{(l)}=W_V^{(l)}h_j^{(l-1)}.
\]

Consequently, two inputs identical through j but different only after j have identical K/V at j in exact arithmetic. RoPE changes position dependence, not the available content. This follows by induction from causal attention and position-wise operations; it is an architectural identity, not a new empirical discovery. The Transformer paper explicitly masks subsequent decoder positions. Its encoder-decoder model also has encoder cross-attention, so the formula here is specifically for the present decoder-only observer, not every architecture. [Vaswani et al., original paper §3.1–3.2](https://arxiv.org/html/1706.03762v7).

Important distinctions for the parent-supplied “300 ... BC” case:

- “Cannot depend on the realized later BC” does not mean “cannot predict an era from earlier words or learned priors.”
- The statement presumes BC occurs in a distinct later physical token; a single token containing both numeral and suffix would require a different position argument. This audit did not inspect the data or tokenizer output.
- A later state can access both numeral and suffix. Access does not prove that it encodes the required temporal relation, retains it, or contributes to the answer.
- The answer can attend directly to later source positions. The numeral's limitation is not a proof that the model as a whole lacks the era information.
- Teacher-forced response states can contain information supplied by the erroneous response prefix itself. Source evidence and answer echo require separate interpretations.

## Verified prior work

### Echo embeddings: exact failure mode already explicit

Jacob Mitchell Springer and colleagues, *Repetition Improves Language Model Embeddings*: first arXiv submission **2024-02-23**, version 2 **2025-09-07**; formal venue **ICLR 2025**, not ICML 2024. Section 3.1 explains that early contextual embeddings cannot include later tokens, causing mean pooling to dilute suffix distinctions; last-token pooling has a different recency problem. Section 3.2 repeats the input and extracts the repeated occurrence, which can read the complete first occurrence without changing causal attention. The evaluated zero-shot backbones include Mistral-7B-Instruct-v0.1, Llama-2-7B-chat and Sheared-Llama-1.3B; this is embedding evaluation, not hallucination detection. The earlier and revised abstracts report different improvement summaries, so this audit does not mix their percentages. The relevant precedent is the information-access argument and repetition, not a transferable detector result. [Paper, §3 and Appendix A.2](https://arxiv.org/html/2402.15449v2), [official ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e9b0c9480b03f9a6bca24ed8c08ee058-Abstract-Conference.html), [version history](https://arxiv.org/abs/2402.15449).

### LLM2Vec: bidirectional adaptation, not the unchanged generator

Parishad BehnamGhader and colleagues: first submitted **2024-04-09**, revised **2024-08-21**, **COLM 2024**. The three steps are bidirectional attention, masked next-token prediction, and unsupervised contrastive learning. Section 4.1 explicitly tests future-token information using shared-prefix sentence comparisons. The four backbones are Sheared-Llama-1.3B, Llama-2-7B, Mistral-7B and Meta-Llama-3-8B. Simply removing the mask can worsen embedding performance; the adaptation is material. This provides a known solution to incomplete contextualization, but changes computation and normally weights/objective, so its resulting states are not the original observer's causal K/V. Its unsupervised adaptation should also be distinguished from its separately reported supervised contrastive variant. It does not establish a hallucination detector based on source numeral closure. [Paper, §2.1–2.2 and §4.1](https://arxiv.org/html/2404.05961v2), [dates and venue](https://arxiv.org/abs/2404.05961).

### Binding IDs: the entity-following token is already part of the carrier

Jiahai Feng and Jacob Steinhardt: first submitted **2023-10-26**, revised **2024-05-06**, **ICLR 2024**. The paper uses controlled entity–attribute tasks and interventions on activations, primarily LLaMA-30B, with broader Pythia/LLaMA family analysis. Crucially, §3.1 says both the entity token and the immediately subsequent token encode binding information; its entity activation definition includes both. Thus “the relevant representation need not sit entirely on the literal entity token” is already explicit. The paper tests binding-ID factorization and position independence; it does not identify arbitrary natural-language sentence closures, an era-specific source K/V failure, or an unsupervised detector on original responses. Its intervention evidence cannot be replaced by one attention heatmap. [Paper, §3.1 and §5](https://arxiv.org/html/2310.17191v2), [official venue](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9d1b7fc578c0d2d6431fc26d736ecaf3-Abstract-Conference.html), [version history](https://arxiv.org/abs/2310.17191).

### ReDeEP: internal semantic evidence and a chunk alternative already exist

Zhongxiang Sun and colleagues: first submitted **2024-10-15**, revised **2025-01-21**, **ICLR 2025**. ReDeEP(token)'s external-context score uses cosine similarity between the response token's final hidden state and a mean of final hidden states of high-attention context tokens; it combines this with an FFN-based parametric score. Its empirical study begins with Llama2-7B-chat/RAGTruth. Section 4.2 explicitly motivates a chunk version by expense and incomplete contextual consideration; it selects attended chunk pairs and compares embeddings, implemented with BGE-base-en-v1.5. These are substantial precedents for criticizing bare token evidence and comparing broader semantic context. However, the reviewed sections do not isolate a later era qualifier missing from the earlier source K/V, nor prove a closure-specific mediation mechanism. Raw K/V and final-layer semantic hidden states are different objects. [Paper, §3.1, §4.1–4.2 and Appendix I](https://arxiv.org/html/2410.11414v2), [official conference record](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7daf60e805e596c3bd1e843e72ea5560-Abstract-Conference.html).

### Re-reading is also established outside embeddings

Xiaohan Xu and colleagues, *Re-Reading Improves Reasoning in Large Language Models*: arXiv first appeared in **September 2023**; formal venue **EMNLP 2024**. RE2 repeats the question. Its Figure 1–2 discussion directly connects causal tokens' missing later question context to second-copy access to the first complete question. It is a prompting intervention for reasoning, not a hallucination detector, and changes the input. The unchanged natural-case M4 is therefore not a test of RE2 and should not silently become one. [Official paper and venue](https://aclanthology.org/2024.emnlp-main.871/), [paper §1–2](https://arxiv.org/html/2309.06275v4).

### Token-level repetition and direct hallucination-localization prior

Matija Luka Kukić and colleagues, *Sequence Repetition Enhances Token Embeddings and Improves Sequence Labeling with Decoder-only Language Models*: **2026-01-24** arXiv, **Findings of EACL 2026**, March. It studies repetition for token labeling with fine-tuned decoder classifiers, including Gemma, Qwen3 and Mistral families, and intermediate-layer features. This extends the prior beyond pooled sentence embeddings. [Official proceedings](https://aclanthology.org/2026.findings-eacl.339/), [paper §2–3](https://arxiv.org/html/2601.17585v1).

More directly, Mikhail Kulakov, Ruslan Solikhov and Ivan Bondarenko, *Sckwoky at SHROOM-Visions*, appeared in the **Third Workshop on Uncertainty-Aware NLP, 2026-10-29**. Its authoritative PDF explicitly identifies causal answer tokens' missing right context, repeats the candidate answer, reads the second copy, and predicts character hallucination spans. It uses a Gemma-4-12B-it backbone with trained LoRA adapters and supervised locator/type classifiers. This is multimodal, label-trained detection rather than our frozen unsupervised natural-text setting, but it defeats a broad claim that repetition/right-context token features for hallucination localization are new. The PDF and webpage abstract differ slightly in reported IoU; no performance number is imported here. [Authoritative PDF, abstract and §1](https://aclanthology.org/2026.uncertainlp-main.61.pdf), [conference metadata](https://aclanthology.org/2026.uncertainlp-main.61/).

## What is and is not answered for M4

| Question | Evidence status before M4 |
|---|---|
| Can a strict causal numeral K/V depend on a later realized era token? | No; known architecture identity, conditional on the exact token axis. |
| Can later or repeated states gain access to the complete phrase? | Yes, access is known; useful semantic encoding is empirical. |
| Has right-context-aware token representation been used for hallucination localization? | Yes, a directly relevant supervised multimodal precedent exists. |
| Has this audit established that R15716's error is caused by numeral-only retrieval rather than era semantics, retrieval/readout failure, or answer-prefix echo? | No. |
| Does source closure outperform token evidence or direct verification for natural original-response hallucination detection at fixed coverage/cost? | Not established by these readings or the proposed one-case observation. |
| Is the exact source-closure diagnostic novel? | Unknown; this bounded audit is not an exhaustive novelty search. |

The remaining discriminator is a **case-specific information-and-use question**, not whether causal masking exists: in the unchanged observer trace, which actual source positions are read while the original response expresses the numeral and era, and is the available evidence confined to an incomplete prefix or also present at later complete-source positions? Attention and hidden-state observations can expose compatibility with these alternatives. They cannot alone show which information caused the error, whether a later state encodes the relevant relation, or whether any future score would generalize. A matched semantic intervention and frozen natural detection comparison would be separate, unapproved scientific claims; this audit proposes no algorithm or additional model calls.

M4's parent-specified scope remains **one unchanged natural case, 441 tokens, one model call and one original-output head call, all original-response rows and all 32 heads**, subject to its own admission/claim. No repair prompt, retrieved QA, answer replacement, head selection, or target-label use is licensed by this review. The source-only G03 license and the quoted R15716 mismatch are task context supplied by the parent, not independently relabeled by this audit.

## Search coverage and exclusions

Checked primary arXiv full text/version histories, ICLR proceedings, ACL Anthology records and authoritative PDFs on 2026-11-03. Queries covered Echo/LLM2Vec causal embedding limitations, Binding IDs following-token positions, ReDeEP token/chunk context, re-reading, and right-context/repetition hallucination detection. Broad search results using unrelated philosophical or discourse “closure” were not treated as evidence for the specific K/V claim. No negative search result is an absence proof. No paper's aggregate benchmark performance is imported as an expected outcome for this project.
