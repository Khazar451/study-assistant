# Feedback on Nebius Token Factory and NVIDIA Nemotron

Written from building and running this project. Edit freely before pasting into the Devpost feedback field.

## What worked well
- **OpenAI-compatible endpoint.** The stock `openai` client worked against `https://api.tokenfactory.nebius.com/v1/` with only a base URL and key change. Moving from NVIDIA NIM to Token Factory touched one provider-resolution function (`src/providers.py`).
- **One catalog, several Nemotron sizes.** Nano (30B-A3B), Super (120B-A12B) and Ultra (550B-A55B) behind a single key let us route by task: Nano for query rewriting and reranking, Super for everyday tutoring, Ultra for derivations and proofs.
- **Public model catalog with prices and throughput** made it easy to choose a model per pipeline stage without trial and error.

## Friction we hit
1. **Reasoning tokens consume `max_tokens`.** On short helper calls (query rewrite, rerank) Nemotron spent the whole budget thinking and returned empty content. We now pass `chat_template_kwargs: {"enable_thinking": false}` for the fast role. A documented, first-class `reasoning` parameter (or a clear note in the model card) would save time.
2. **Inline `<think>` blocks in streamed output.** Tags can be split across stream chunks, so we wrote a streaming filter (`filter_reasoning_stream`). Returning reasoning in a separate field (`reasoning_content`) for streaming too would remove the need.
3. **Citation brackets.** Nemotron frequently emitted full-width brackets (【1】) instead of `[1]`, which broke citation parsing; we normalize them. A system-prompt note in the model card about this tendency would help.
4. **Model IDs vs. display names.** The catalog shows "Nemotron-3-Ultra-550b-a55b" and the IDs mix casing: `nvidia/nemotron-3-super-120b-a12b` (lowercase), `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` and `nvidia/Nemotron-3-Ultra-550b-a55b` (mixed). We guessed the Ultra ID from the Super pattern and got `404 model_not_found`; listing `GET /v1/models` showed the real one. A copy-ID button and consistent casing would avoid this.
5. **Embeddings and vision.** Our retrieval embeddings (`llama-nemotron-embed-vl-1b-v2`) and slide vision still run on NVIDIA NIM. Having Nemotron embedding and vision models on Token Factory would let the whole stack run on one provider and one bill.

## Suggestions
- Per-request cost and token breakdown (including reasoning tokens) in the response `usage` object.
- A rate-limit headers contract (`x-ratelimit-remaining`, `retry-after`) documented for public endpoints.
- Example notebooks for RAG with Nemotron that show the reasoning toggle and the citation-format behavior.
6. **Intermittent stalls on the NVIDIA-hosted embedding endpoint.** Calls to `nvidia/llama-nemotron-embed-vl-1b-v2` on `integrate.api.nvidia.com` usually return in 0.2 to 0.5 s, but a large share of identical requests hang until the client times out. We now use a 4 s timeout with retries for single queries, which cut a typical query from 13 to 38 s down to about 4 s. A hosted Nemotron embedding model on Token Factory would remove this dependency.
