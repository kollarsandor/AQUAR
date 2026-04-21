import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader


def compute_perplexity(model, dataloader, device, max_recurrence=1, tokenizer=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
            else:
                input_ids = batch.to(device)
                attention_mask = None
            B, S = input_ids.shape
            input_ids_cut = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            mask = None
            if attention_mask is not None:
                mask = attention_mask[:, 1:]
            logits = model(input_ids_cut, max_recurrence=max_recurrence)
            if isinstance(logits, tuple):
                logits = logits[0]
            V = logits.size(-1)
            logits_flat = logits.reshape(-1, V)
            targets_flat = targets.reshape(-1)
            loss_flat = F.cross_entropy(logits_flat, targets_flat, reduction="none")
            if mask is not None:
                mask_flat = mask.reshape(-1).float()
                loss_flat = loss_flat * mask_flat
                n_tokens = mask_flat.sum().item()
            else:
                n_tokens = targets_flat.numel()
            total_loss += loss_flat.sum().item()
            total_tokens += n_tokens
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 100 else float("inf")
    return ppl, avg_loss


def _tokenize_wikitext(example, tokenizer, seq_len, stride):
    token_ids = tokenizer.encode(example["text"])
    if len(token_ids) < 2:
        return []
    result = []
    for start in range(0, len(token_ids) - 1, stride):
        end = min(start + seq_len, len(token_ids))
        chunk = token_ids[start:end]
        if len(chunk) >= 2:
            result.append({"input_ids": chunk, "attention_mask": [1] * len(chunk)})
        if end >= len(token_ids):
            break
    return result


def _wikitext_text_filter(example):
    text = example["text"].strip()
    return len(text) > 100 and not text.startswith("=")


def compute_wikitext_perplexity(model, tokenizer, device, max_recurrence=1, seq_len=2048):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    filtered = dataset.filter(_wikitext_text_filter)
    stride = seq_len // 2
    all_examples = []
    for example in filtered:
        chunks = _tokenize_wikitext(example, tokenizer, seq_len, stride)
        all_examples.extend(chunks)
    if not all_examples:
        return float("inf"), float("inf")

    def collate_dynamic(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        attention_mask = []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    dataloader = DataLoader(all_examples, batch_size=4, shuffle=False, collate_fn=collate_dynamic)
    ppl, loss = compute_perplexity(model, dataloader, device, max_recurrence=max_recurrence, tokenizer=tokenizer)
    return ppl, loss


def compute_lambada_perplexity(model, tokenizer, device, max_recurrence=1):
    dataset = load_dataset("lambada", split="validation")
    correct = 0
    total = 0
    with torch.no_grad():
        for example in dataset:
            text = example["text"].strip()
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            context_tokens = tokens[:-1]
            last_token = tokens[-1]
            last_word = text.split()[-1]
            input_tensor = torch.tensor([context_tokens], dtype=torch.long, device=device)
            logits = model(input_tensor, max_recurrence=max_recurrence)
            if isinstance(logits, tuple):
                logits = logits[0]
            last_logits = logits[0, -1, :]
            pred_token = torch.argmax(last_logits).item()
            pred_word = tokenizer.decode([pred_token]).strip().lower()
            target_word = last_word.strip().lower()
            if pred_word == target_word:
                correct += 1
            total += 1
    accuracy = correct / total if total > 0 else 0.0
    loss_val = 1.0 - accuracy
    ppl = torch.exp(torch.tensor(loss_val)).item() if loss_val >= 0 else float("inf")
    return ppl, loss_val
