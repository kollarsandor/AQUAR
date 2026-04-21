import random
import torch
import torch.nn.functional as F
from datasets import load_dataset
from abc import ABC, abstractmethod


class BaseTask(ABC):
    def __init__(self, name, task_type, num_shots, category, is_core=False, is_core_ext=False):
        self.name = name
        self.task_type = task_type
        self.num_shots = num_shots
        self.category = category
        self.is_core = is_core
        self.is_core_ext = is_core_ext
        self.data = None

    @abstractmethod
    def load_data(self):
        return None

    @abstractmethod
    def evaluate(self, model, tokenizer, device, max_recurrence=1, seed=0):
        return 0.0

    @abstractmethod
    def format_prompt(self, example, few_shot_examples):
        return "", [], 0

    @staticmethod
    def mc_score(model, tokenizer, prompt, choices, device, max_recurrence=1):
        model.eval()
        nlls = []
        for choice in choices:
            full_text = prompt + " " + choice
            tokens = tokenizer.encode(full_text)
            prompt_tokens = tokenizer.encode(prompt)
            if len(tokens) <= len(prompt_tokens):
                nlls.append(1e9)
                continue
            answer_tokens = tokens[len(prompt_tokens):]
            input_tensor = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_tensor, max_recurrence=max_recurrence)
                if isinstance(logits, tuple):
                    logits = logits[0]
            answer_len = len(answer_tokens)
            relevant_logits = logits[0, -(answer_len):, :]
            relevant_targets = torch.tensor(answer_tokens, dtype=torch.long, device=device)
            loss = F.cross_entropy(relevant_logits, relevant_targets, reduction="mean")
            nlls.append(loss.item())
        return min(nlls)

    @staticmethod
    def lm_score(model, tokenizer, prompt, expected, device, max_recurrence=1):
        model.eval()
        full_text = prompt + expected
        tokens = tokenizer.encode(full_text)
        prompt_tokens = tokenizer.encode(prompt)
        if len(tokens) <= len(prompt_tokens):
            return 0.0
        answer_tokens = tokens[len(prompt_tokens):]
        input_tensor = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_tensor, max_recurrence=max_recurrence)
            if isinstance(logits, tuple):
                logits = logits[0]
        answer_len = len(answer_tokens)
        relevant_logits = logits[0, -(answer_len):, :]
        generated = torch.argmax(relevant_logits, dim=-1).tolist()
        expected_ids = answer_tokens
        matches = sum(1 for g, e in zip(generated, expected_ids) if g == e)
        return 1.0 if matches == len(expected_ids) else 0.0

    @staticmethod
    def schema_score(model, tokenizer, prompt, schema_fn, device, max_recurrence=1):
        model.eval()
        options = schema_fn()
        nlls = []
        for label, completion in options:
            full_text = prompt + " " + completion
            tokens = tokenizer.encode(full_text)
            prompt_tokens = tokenizer.encode(prompt)
            if len(tokens) <= len(prompt_tokens):
                nlls.append(1e9)
                continue
            answer_tokens = tokens[len(prompt_tokens):]
            input_tensor = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_tensor, max_recurrence=max_recurrence)
                if isinstance(logits, tuple):
                    logits = logits[0]
            answer_len = len(answer_tokens)
            relevant_logits = logits[0, -(answer_len):, :]
            relevant_targets = torch.tensor(answer_tokens, dtype=torch.long, device=device)
            loss = F.cross_entropy(relevant_logits, relevant_targets, reduction="mean")
            nlls.append(loss.item())
        best_idx = int(torch.argmin(torch.tensor(nlls)).item())
        return options[best_idx][0]


class MultipleChoiceTask(BaseTask):
    def evaluate(self, model, tokenizer, device, max_recurrence=1, seed=0):
        if self.data is None:
            self.load_data()
        rng = random.Random(seed)
        examples = list(self.data)
        rng.shuffle(examples)
        if self.num_shots > 0:
            few_shot_pool = examples[: max(20, self.num_shots * 3)]
            eval_examples = examples[max(20, self.num_shots * 3):]
        else:
            few_shot_pool = []
            eval_examples = examples
        scores = []
        for example in eval_examples:
            few_shot = rng.sample(few_shot_pool, min(self.num_shots, len(few_shot_pool))) if few_shot_pool else []
            prompt, choices, label = self.format_prompt(example, few_shot)
            nlls = []
            for choice in choices:
                full_text = prompt + " " + choice
                tokens = tokenizer.encode(full_text)
                prompt_tok = tokenizer.encode(prompt)
                if len(tokens) <= len(prompt_tok):
                    nlls.append(1e9)
                    continue
                answer_tokens = tokens[len(prompt_tok):]
                input_tensor = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = model(input_tensor, max_recurrence=max_recurrence)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                alen = len(answer_tokens)
                rlogits = logits[0, -(alen):, :]
                rtargets = torch.tensor(answer_tokens, dtype=torch.long, device=device)
                loss = F.cross_entropy(rlogits, rtargets, reduction="mean")
                nlls.append(loss.item())
            pred = int(torch.argmin(torch.tensor(nlls)).item())
            if pred == label:
                scores.append(1.0)
            else:
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0


class LanguageModelTask(BaseTask):
    def evaluate(self, model, tokenizer, device, max_recurrence=1, seed=0):
        if self.data is None:
            self.load_data()
        rng = random.Random(seed)
        examples = list(self.data)
        rng.shuffle(examples)
        if self.num_shots > 0:
            few_shot_pool = examples[: max(20, self.num_shots * 3)]
            eval_examples = examples[max(20, self.num_shots * 3):]
        else:
            few_shot_pool = []
            eval_examples = examples
        scores = []
        for example in eval_examples:
            few_shot = rng.sample(few_shot_pool, min(self.num_shots, len(few_shot_pool))) if few_shot_pool else []
            prompt, expected, _ = self.format_prompt(example, few_shot)
            full_text = prompt + expected
            tokens = tokenizer.encode(full_text)
            prompt_tok = tokenizer.encode(prompt)
            if len(tokens) <= len(prompt_tok):
                scores.append(0.0)
                continue
            answer_tokens = tokens[len(prompt_tok):]
            input_tensor = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_tensor, max_recurrence=max_recurrence)
                if isinstance(logits, tuple):
                    logits = logits[0]
            alen = len(answer_tokens)
            rlogits = logits[0, -(alen):, :]
            generated = torch.argmax(rlogits, dim=-1).tolist()
            matches = sum(1 for g, e in zip(generated, answer_tokens) if g == e)
            if matches == len(answer_tokens):
                scores.append(1.0)
            else:
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0


class SchemaTask(BaseTask):
    def evaluate(self, model, tokenizer, device, max_recurrence=1, seed=0):
        if self.data is None:
            self.load_data()
        rng = random.Random(seed)
        examples = list(self.data)
        rng.shuffle(examples)
        if self.num_shots > 0:
            few_shot_pool = examples[: max(20, self.num_shots * 3)]
            eval_examples = examples[max(20, self.num_shots * 3):]
        else:
            few_shot_pool = []
            eval_examples = examples
        scores = []
        for example in eval_examples:
            few_shot = rng.sample(few_shot_pool, min(self.num_shots, len(few_shot_pool))) if few_shot_pool else []
            prompt, schema_fn, label = self.format_prompt(example, few_shot)
            options = schema_fn()
            nlls = []
            for lbl, completion in options:
                full_text = prompt + " " + completion
                tokens = tokenizer.encode(full_text)
                prompt_tok = tokenizer.encode(prompt)
                if len(tokens) <= len(prompt_tok):
                    nlls.append(1e9)
                    continue
                answer_tokens = tokens[len(prompt_tok):]
                input_tensor = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = model(input_tensor, max_recurrence=max_recurrence)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                alen = len(answer_tokens)
                rlogits = logits[0, -(alen):, :]
                rtargets = torch.tensor(answer_tokens, dtype=torch.long, device=device)
                loss = F.cross_entropy(rlogits, rtargets, reduction="mean")
                nlls.append(loss.item())
            best_idx = int(torch.argmin(torch.tensor(nlls)).item())
            if best_idx == label:
                scores.append(1.0)
            else:
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0
