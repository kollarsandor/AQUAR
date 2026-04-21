import json
import random
import bisect
from typing import List, Optional, Dict, Any, Iterator

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class BestFitCropPackedDataset(Dataset):
    def __init__(
        self,
        tokenized_documents: List[List[int]],
        seq_len: int = 2048,
        bos_id: int = 0,
        eos_id: int = 1,
        pad_id: int = 2,
    ):
        self.seq_len = seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        processed_docs = []
        for doc in tokenized_documents:
            tokens = list(doc)
            if len(tokens) == 0:
                continue
            if tokens[0] != bos_id:
                tokens = [bos_id] + tokens
            if tokens[-1] != eos_id:
                tokens = tokens + [eos_id]
            if len(tokens) <= seq_len:
                processed_docs.append(tokens)
            else:
                offset = 0
                while offset < len(tokens):
                    end = offset + seq_len
                    if end >= len(tokens):
                        chunk = tokens[offset:]
                        if chunk[0] != bos_id:
                            chunk = [bos_id] + chunk
                        if len(chunk) <= seq_len:
                            chunk = chunk + [eos_id] if chunk[-1] != eos_id else chunk
                            if len(chunk) <= seq_len:
                                processed_docs.append(chunk)
                        break
                    else:
                        chunk = tokens[offset:end]
                        if chunk[0] != bos_id:
                            chunk = [bos_id] + chunk
                        if len(chunk) <= seq_len:
                            if chunk[-1] != eos_id:
                                chunk = chunk + [eos_id]
                        if len(chunk) > seq_len:
                            chunk = chunk[:seq_len]
                        processed_docs.append(chunk)
                        offset = end

        self.documents = processed_docs
        self.doc_lengths = [len(d) for d in self.documents]

        sorted_indices = sorted(range(len(self.documents)), key=lambda i: -self.doc_lengths[i])
        sorted_lengths = [self.doc_lengths[i] for i in sorted_indices]

        self.packed_sequences = []
        used = set()
        for start in range(len(sorted_indices)):
            if start in used:
                continue
            idx = sorted_indices[start]
            if self.doc_lengths[idx] > seq_len:
                used.add(start)
                self.packed_sequences.append((self.documents[idx],))
                continue
            current_len = self.doc_lengths[idx]
            selected = [start]
            for j in range(start + 1, len(sorted_indices)):
                if j in used:
                    continue
                j_idx = sorted_indices[j]
                if current_len + self.doc_lengths[j_idx] <= seq_len:
                    selected.append(j)
                    current_len += self.doc_lengths[j_idx]
                    if current_len >= seq_len:
                        break
            for s in selected:
                used.add(s)
            if len(selected) == 1:
                idx_s = sorted_indices[selected[0]]
                doc_tokens = list(self.documents[idx_s])
                if len(doc_tokens) < seq_len:
                    doc_tokens = doc_tokens + [pad_id] * (seq_len - len(doc_tokens))
                self.packed_sequences.append((doc_tokens,))
            else:
                combined = []
                for s in selected:
                    combined.extend(self.documents[sorted_indices[s]])
                if len(combined) > seq_len:
                    combined = combined[:seq_len]
                elif len(combined) < seq_len:
                    combined = combined + [pad_id] * (seq_len - len(combined))
                self.packed_sequences.append((combined,))

    def __len__(self) -> int:
        return len(self.packed_sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.packed_sequences[idx][0], dtype=torch.long)


class SequencePackingCollator:
    def __init__(self, pad_id: int = 2, seq_len: int = 2048):
        self.pad_id = pad_id
        self.seq_len = seq_len

    def __call__(self, batch: List[torch.Tensor]) -> torch.Tensor:
        result = torch.full((len(batch), self.seq_len), self.pad_id, dtype=torch.long)
        for i, seq in enumerate(batch):
            length = min(seq.size(0), self.seq_len)
            result[i, :length] = seq[:length]
        return result


class StandardTokenDataset(Dataset):
    def __init__(self, token_ids: List[int], seq_len: int = 2048):
        self.seq_len = seq_len
        self.num_sequences = max(1, (len(token_ids) - 1) // seq_len)
        self.token_ids = token_ids[:self.num_sequences * seq_len + 1]

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        tokens = self.token_ids[start:end]
        if len(tokens) < self.seq_len + 1:
            tokens = tokens + [0] * (self.seq_len + 1 - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)


def _load_jsonl(path: str) -> List[str]:
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and 'text' in obj:
                texts.append(obj['text'])
            elif isinstance(obj, str):
                texts.append(obj)
    return texts


def _load_raw_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _tokenize_documents(texts: List[str], tokenizer) -> List[List[int]]:
    documents = []
    for text in texts:
        if isinstance(tokenizer, dict):
            tokens = []
            for ch in text:
                tokens.append(tokenizer.get(ch, tokenizer.get('<unk>', 0)))
            documents.append(tokens)
        else:
            tokens = tokenizer.encode(text)
            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()
            documents.append(tokens)
    return documents


def _tokenize_raw(text: str, tokenizer) -> List[int]:
    if isinstance(tokenizer, dict):
        return [tokenizer.get(ch, tokenizer.get('<unk>', 0)) for ch in text]
    else:
        tokens = tokenizer.encode(text)
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        return tokens


def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: Any,
    batch_size: int,
    seq_len: int = 2048,
    num_workers: int = 4,
    distributed: bool = False,
    pack_documents: bool = True,
    bos_id: int = 0,
    eos_id: int = 1,
    pad_id: int = 2,
    seed: int = 42,
) -> tuple:
    if train_path.endswith('.jsonl'):
        train_texts = _load_jsonl(train_path)
    else:
        train_text = _load_raw_text(train_path)
        train_texts = [train_text]

    if val_path.endswith('.jsonl'):
        val_texts = _load_jsonl(val_path)
    else:
        val_text = _load_raw_text(val_path)
        val_texts = [val_text]

    if pack_documents:
        train_docs = _tokenize_documents(train_texts, tokenizer)
        val_docs = _tokenize_documents(val_texts, tokenizer)

        train_dataset = BestFitCropPackedDataset(
            train_docs, seq_len=seq_len, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id
        )
        val_dataset = BestFitCropPackedDataset(
            val_docs, seq_len=seq_len, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id
        )

        train_collator = SequencePackingCollator(pad_id=pad_id, seq_len=seq_len)
        val_collator = SequencePackingCollator(pad_id=pad_id, seq_len=seq_len)
    else:
        all_train_tokens = []
        for text in train_texts:
            all_train_tokens.extend(_tokenize_raw(text, tokenizer))
        all_val_tokens = []
        for text in val_texts:
            all_val_tokens.extend(_tokenize_raw(text, tokenizer))

        train_dataset = StandardTokenDataset(all_train_tokens, seq_len=seq_len)
        val_dataset = StandardTokenDataset(all_val_tokens, seq_len=seq_len)

        train_collator = None
        val_collator = None

    train_sampler: Optional[DistributedSampler] = None
    val_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset, shuffle=True, seed=seed, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset, shuffle=False, seed=seed, drop_last=False
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_collator,
        drop_last=(train_sampler is None),
        persistent_workers=num_workers > 0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=val_collator,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_dataloader, val_dataloader
