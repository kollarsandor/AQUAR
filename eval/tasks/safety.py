import random
from datasets import load_dataset
from .base import MultipleChoiceTask, LanguageModelTask, SchemaTask


class WinogenderFemale(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="winogender_female",
            task_type="MC",
            num_shots=10,
            category="safety",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("super_glue", "wsc", split="validation")
        self.data = []
        for ex in dataset:
            text = ex["text"]
            span1 = ex.get("span1_text", "")
            span2 = ex.get("span2_text", "")
            label = ex.get("label", -1)
            if label == -1:
                continue
            if "she" in text.lower() or "her" in text.lower() or "woman" in text.lower() or "female" in text.lower() or "girl" in text.lower() or "mother" in text.lower() or "sister" in text.lower():
                candidates = [span1, span2]
                self.data.append({
                    "text": text,
                    "span1": span1,
                    "span2": span2,
                    "candidates": candidates,
                    "label": label,
                })

    def format_prompt(self, example, few_shot_examples):
        text = example["text"]
        candidates = example["candidates"]
        label = example["label"]
        parts = []
        for fs in few_shot_examples:
            parts.append("Passage: " + fs["text"])
            parts.append("What does the pronoun refer to? " + str(fs["candidates"][fs["label"]]))
            parts.append("")
        parts.append("Passage: " + text)
        parts.append("What does the pronoun refer to?")
        prompt = "\n".join(parts)
        choices = candidates
        return prompt, choices, label


class WinogenderMale(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="winogender_male",
            task_type="MC",
            num_shots=10,
            category="safety",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("super_glue", "wsc", split="validation")
        self.data = []
        for ex in dataset:
            text = ex["text"]
            span1 = ex.get("span1_text", "")
            span2 = ex.get("span2_text", "")
            label = ex.get("label", -1)
            if label == -1:
                continue
            if "he" in text.lower() or "him" in text.lower() or "his" in text.lower() or "man" in text.lower() or "male" in text.lower() or "boy" in text.lower() or "father" in text.lower() or "brother" in text.lower():
                candidates = [span1, span2]
                self.data.append({
                    "text": text,
                    "span1": span1,
                    "span2": span2,
                    "candidates": candidates,
                    "label": label,
                })

    def format_prompt(self, example, few_shot_examples):
        text = example["text"]
        candidates = example["candidates"]
        label = example["label"]
        parts = []
        for fs in few_shot_examples:
            parts.append("Passage: " + fs["text"])
            parts.append("What does the pronoun refer to? " + str(fs["candidates"][fs["label"]]))
            parts.append("")
        parts.append("Passage: " + text)
        parts.append("What does the pronoun refer to?")
        prompt = "\n".join(parts)
        choices = candidates
        return prompt, choices, label


class EnterprisePIIClassification(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="enterprise_pii",
            task_type="MC",
            num_shots=10,
            category="safety",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("ai4privacy/pii-masking-200k", split="train")
        self.data = []
        pii_types = ["NAME", "EMAIL", "PHONE", "ADDRESS", "SSN", "CREDIT_CARD", "DATE_OF_BIRTH", "PASSPORT"]
        count = 0
        for ex in dataset:
            source_text = ex.get("source_text", "")
            masked_text = ex.get("masked_text", source_text)
            privacy_mask = ex.get("privacy_mask", "[]")
            if not source_text:
                continue
            has_pii = False
            for pii_type in pii_types:
                if pii_type in privacy_mask:
                    has_pii = True
                    break
            self.data.append({
                "source_text": source_text,
                "masked_text": masked_text,
                "has_pii": has_pii,
                "pii_mask": privacy_mask,
            })
            count += 1
            if count >= 2000:
                break

    def format_prompt(self, example, few_shot_examples):
        source = example["source_text"]
        has_pii = example["has_pii"]
        parts = []
        for fs in few_shot_examples:
            parts.append("Text: " + fs["source_text"][:200])
            ans = "Contains PII" if fs["has_pii"] else "No PII"
            parts.append("Classification: " + ans)
            parts.append("")
        parts.append("Text: " + source[:200])
        parts.append("Classification:")
        prompt = "\n".join(parts)
        choices = ["Contains PII", "No PII"]
        label = 0 if has_pii else 1
        return prompt, choices, label


class BBQTask(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="bbq",
            task_type="MC",
            num_shots=3,
            category="safety",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("mllm/BBQ", split="test")
        self.data = []
        for ex in dataset:
            context = ex.get("context", "")
            question = ex.get("question", "")
            ans0 = ex.get("ans0", "")
            ans1 = ex.get("ans1", "")
            ans2 = ex.get("ans2", "")
            label_str = ex.get("label", "0")
            label_idx = int(label_str)
            condition = ex.get("condition", "")
            category = ex.get("category", "")
            self.data.append({
                "context": context,
                "question": question,
                "ans0": ans0,
                "ans1": ans1,
                "ans2": ans2,
                "label": label_idx,
                "condition": condition,
                "category": category,
            })

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            parts.append("Context: " + fs["context"])
            parts.append("Question: " + fs["question"])
            fs_choices = [fs["ans0"], fs["ans1"], fs["ans2"]]
            parts.append("Answer: " + fs_choices[fs["label"]])
            parts.append("")
        parts.append("Context: " + example["context"])
        parts.append("Question: " + example["question"])
        parts.append("Answer:")
        prompt = "\n".join(parts)
        choices = [example["ans0"], example["ans1"], example["ans2"]]
        return prompt, choices, example["label"]
