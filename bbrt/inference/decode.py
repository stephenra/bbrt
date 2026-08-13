"""Decoding strategies for the Transformer seq2seq.

Replaces the OpenNMT ``build_translator`` path in the original ``translate.py``
with direct greedy / top-k-top-p sampling / beam search over our model. All
methods take a list of (space-tokenized or raw) SELFIES source strings and
return, per source, a list of decoded SELFIES strings (concatenated, ready for
``selfies.decoder``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from bbrt.data.collate import pad_sequences
from bbrt.data.tokenizer import SelfiesTokenizer
from bbrt.models.transformer import TransformerSeq2Seq


def _filter_top_k_top_p(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Mask logits (B, V) to the top-k / nucleus-p set (in place-safe)."""
    logits = logits.clone()
    if top_k and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 0] = False  # always keep the most likely token
        remove = torch.zeros_like(remove).scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove, float("-inf"))
    return logits


class Generator:
    """Batched autoregressive decoder around a trained model.

    Note: decoding recomputes the decoder over the growing prefix each step
    (no KV cache). SELFIES sequences are short, so this stays fast; a KV cache
    is a straightforward future optimization.
    """

    def __init__(
        self,
        model: TransformerSeq2Seq,
        tokenizer: SelfiesTokenizer,
        device: str | torch.device = "cpu",
    ):
        self.model = model.to(device).eval()
        self.tok = tokenizer
        self.device = torch.device(device)

    # -- shared encoder ----------------------------------------------------- #
    @torch.no_grad()
    def _encode(self, src_selfies: list[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = [self.tok.encode(s)[:max_len] for s in src_selfies]
        src = pad_sequences(seqs, self.tok.pad_id).to(self.device)
        src_pad_mask = src.eq(self.tok.pad_id)
        memory = self.model.encode(src, src_pad_mask)
        return memory, src_pad_mask

    def _ids_to_selfies(self, ids: torch.Tensor) -> str:
        """Decode a 1-D id tensor to a concatenated SELFIES string (stops at EOS)."""
        return self.tok.decode(ids.tolist())

    # -- top-level API ------------------------------------------------------ #
    @torch.no_grad()
    def translate(
        self,
        src_selfies: list[str],
        mode: str = "sd",
        n_best: int = 5,
        top_k: int = 5,
        top_p: float = 1.0,
        temperature: float = 1.0,
        beam_size: int = 10,
        max_len: int = 256,
        seed: int | None = None,
    ) -> list[list[str]]:
        if mode == "greedy":
            return self.greedy(src_selfies, max_len=max_len)
        if mode == "beam":
            return self.beam_search(
                src_selfies, beam_size=beam_size, n_best=n_best, max_len=max_len
            )
        if mode == "sd":
            return self.sample(
                src_selfies,
                n_best=n_best,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_len=max_len,
                seed=seed,
            )
        raise ValueError(f"unknown decode mode {mode!r}")

    @staticmethod
    def _reorder_caches(caches: list[dict], index: torch.Tensor) -> None:
        """Reindex per-layer KV caches along the batch dim (for beam reordering)."""
        for cache in caches:
            for name in ("self", "cross"):
                if name in cache:
                    k, v = cache[name]
                    cache[name] = (k.index_select(0, index), v.index_select(0, index))

    # -- greedy ------------------------------------------------------------- #
    @torch.no_grad()
    def greedy(self, src_selfies: list[str], max_len: int = 256) -> list[list[str]]:
        """Deterministic greedy decode (one sequence per source)."""
        n_src = len(src_selfies)
        memory, mem_mask = self._encode(src_selfies, max_len)
        caches = self.model.init_caches()
        cur = torch.full((n_src, 1), self.tok.bos_id, dtype=torch.long, device=self.device)
        finished = torch.zeros(n_src, dtype=torch.bool, device=self.device)
        collected: list[torch.Tensor] = []
        for pos in range(max_len):
            logits = self.model.decode_step(cur, memory, mem_mask, caches, pos)[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            nxt[finished] = self.tok.pad_id
            collected.append(nxt)
            finished |= nxt.squeeze(1).eq(self.tok.eos_id)
            cur = nxt
            if bool(finished.all()):
                break
        ys = torch.cat(collected, dim=1)
        return [[self._ids_to_selfies(ys[i])] for i in range(n_src)]

    # -- stochastic top-k / nucleus sampling -------------------------------- #
    @torch.no_grad()
    def sample(
        self,
        src_selfies: list[str],
        n_best: int = 5,
        top_k: int = 5,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_len: int = 256,
        seed: int | None = None,
    ) -> list[list[str]]:
        n_src = len(src_selfies)
        memory, src_mask = self._encode(src_selfies, max_len)
        # Replicate each source n_best times to draw n_best independent samples.
        mem = memory.repeat_interleave(n_best, dim=0)
        mem_mask = src_mask.repeat_interleave(n_best, dim=0)
        b = mem.size(0)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))

        caches = self.model.init_caches()
        cur = torch.full((b, 1), self.tok.bos_id, dtype=torch.long, device=self.device)
        finished = torch.zeros(b, dtype=torch.bool, device=self.device)
        collected: list[torch.Tensor] = []
        for pos in range(max_len):
            logits = self.model.decode_step(cur, mem, mem_mask, caches, pos)[:, -1, :]
            logits = logits / max(temperature, 1e-6)
            logits = _filter_top_k_top_p(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=gen)  # (b, 1)
            nxt[finished] = self.tok.pad_id
            collected.append(nxt)
            finished |= nxt.squeeze(1).eq(self.tok.eos_id)
            cur = nxt
            if bool(finished.all()):
                break

        ys = torch.cat(collected, dim=1)
        out = [self._ids_to_selfies(ys[i]) for i in range(b)]
        return [out[i * n_best : (i + 1) * n_best] for i in range(n_src)]

    # -- beam search -------------------------------------------------------- #
    @torch.no_grad()
    def beam_search(
        self,
        src_selfies: list[str],
        beam_size: int = 10,
        n_best: int = 5,
        max_len: int = 256,
        length_penalty: float = 1.0,
    ) -> list[list[str]]:
        if n_best > beam_size:
            raise ValueError("n_best cannot exceed beam_size")
        k = beam_size
        n_src = len(src_selfies)
        memory, src_mask = self._encode(src_selfies, max_len)
        mem = memory.repeat_interleave(k, dim=0)
        mem_mask = src_mask.repeat_interleave(k, dim=0)
        vocab = self.model.vocab_size
        dev = self.device

        caches = self.model.init_caches()
        cur = torch.full((n_src * k, 1), self.tok.bos_id, dtype=torch.long, device=dev)
        beam_scores = torch.full((n_src, k), float("-inf"), device=dev)
        beam_scores[:, 0] = 0.0
        beam_scores = beam_scores.view(-1)  # (n_src*k,)
        finished = torch.zeros(n_src * k, dtype=torch.bool, device=dev)
        ys: torch.Tensor | None = None  # accumulated tokens (excl. BOS)

        for pos in range(max_len):
            logits = self.model.decode_step(cur, mem, mem_mask, caches, pos)[:, -1, :]
            logp = F.log_softmax(logits, dim=-1)
            if bool(finished.any()):
                # Finished beams may only emit pad, adding zero score (frozen).
                logp[finished] = float("-inf")
                logp[finished, self.tok.pad_id] = 0.0
            next_scores = (beam_scores.unsqueeze(1) + logp).view(n_src, k * vocab)
            top_scores, top_idx = next_scores.topk(k, dim=-1)  # (n_src, k)
            beam_idx = top_idx // vocab
            tok_idx = top_idx % vocab
            global_beam = (torch.arange(n_src, device=dev) * k).unsqueeze(1) + beam_idx
            flat_gb = global_beam.reshape(-1)
            self._reorder_caches(caches, flat_gb)  # follow the surviving beams
            new_tok = tok_idx.reshape(-1, 1)
            ys = new_tok if ys is None else torch.cat([ys[flat_gb], new_tok], dim=1)
            beam_scores = top_scores.reshape(-1)
            finished = finished[flat_gb] | new_tok.squeeze(1).eq(self.tok.eos_id)
            cur = new_tok
            if bool(finished.all()):
                break

        # Rank each source's beams by length-normalized score.
        assert ys is not None
        results: list[list[str]] = []
        ys = ys.view(n_src, k, -1)
        scores = beam_scores.view(n_src, k)
        for b in range(n_src):
            cand = []
            for j in range(k):
                seq = ys[b, j]  # already excludes BOS
                toks = self.tok.decode_tokens(seq.tolist())
                length = max(len(toks), 1)
                cand.append((scores[b, j].item() / (length**length_penalty), "".join(toks)))
            cand.sort(key=lambda x: x[0], reverse=True)
            results.append([c[1] for c in cand[:n_best]])
        return results
