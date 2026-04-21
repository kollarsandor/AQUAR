import torch
from .tasks.task_registry import get_all_tasks, get_core_tasks, get_core_ext_tasks, CORE_TASKS, CORE_EXT_TASKS
from .aggregate import compute_core_score, compute_core_extended_score, compute_category_scores


class TestTimeRecurrenceSweep:
    def __init__(self, model, tokenizer, device, t_range=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.t_range = t_range if t_range is not None else range(1, 25)

    def evaluate_all_t(self, val_dataloader=None, task_subset=None):
        sweep_results = {}
        for t in self.t_range:
            sweep_results[t] = self.evaluate_single_t(t, val_dataloader, task_subset)
        return sweep_results

    def evaluate_single_t(self, t, val_dataloader=None, task_subset=None):
        self.model.eval()
        results = {}
        if task_subset is None:
            task_subset = list(CORE_TASKS)
        all_tasks = get_all_tasks()
        for task_name in task_subset:
            if task_name not in all_tasks:
                continue
            task = all_tasks[task_name]
            task.load_data()
            seed_scores = []
            for seed in range(3):
                score = task.evaluate(
                    self.model, self.tokenizer, self.device,
                    max_recurrence=t, seed=seed,
                )
                seed_scores.append(score)
            avg_score = sum(seed_scores) / len(seed_scores)
            results[task_name] = {
                "score": avg_score,
                "seed_scores": seed_scores,
                "task_type": task.task_type,
                "category": task.category,
            }
        val_loss = None
        val_ppl = None
        if val_dataloader is not None:
            val_ppl, val_loss = self._compute_val_perplexity(val_dataloader, t)
        core_score = compute_core_score(results) if results else 0.0
        core_ext_score = compute_core_extended_score(results) if results else 0.0
        cat_scores = compute_category_scores(results) if results else {}
        return {
            "T": t,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "core_score": core_score,
            "core_ext_score": core_ext_score,
            "category_scores": cat_scores,
            "per_task_scores": results,
        }

    def _compute_val_perplexity(self, val_dataloader, t):
        import torch.nn.functional as F
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in val_dataloader:
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                else:
                    input_ids = batch.to(self.device)
                    attention_mask = None
                input_ids_cut = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                logits = self.model(input_ids_cut, max_recurrence=t)
                if isinstance(logits, tuple):
                    logits = logits[0]
                V = logits.size(-1)
                logits_flat = logits.reshape(-1, V)
                targets_flat = targets.reshape(-1)
                loss_flat = F.cross_entropy(logits_flat, targets_flat, reduction="none")
                if attention_mask is not None:
                    mask_flat = attention_mask[:, 1:].reshape(-1).float()
                    loss_flat = loss_flat * mask_flat
                    n_tokens = mask_flat.sum().item()
                else:
                    n_tokens = targets_flat.numel()
                total_loss += loss_flat.sum().item()
                total_tokens += n_tokens
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        ppl = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 100 else float("inf")
        return ppl, avg_loss
