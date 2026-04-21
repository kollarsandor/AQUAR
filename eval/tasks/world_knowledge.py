import random
from datasets import load_dataset
from .base import MultipleChoiceTask, LanguageModelTask, SchemaTask


class JeopardyTask(LanguageModelTask):
    def __init__(self):
        super().__init__(
            name="jeopardy",
            task_type="LM",
            num_shots=10,
            category="world_knowledge",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("jeopardy_all", split="train")
        self.data = []
        for ex in dataset:
            question = ex.get("question", "").strip()
            answer = ex.get("answer", "").strip()
            category = ex.get("category", "").strip()
            if not question or not answer:
                continue
            self.data.append({
                "question": question,
                "answer": answer,
                "category": category,
            })
        random.Random(42).shuffle(self.data)
        self.data = self.data[:2000]

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            parts.append("Category: " + fs["category"])
            parts.append("Question: " + fs["question"])
            parts.append("Answer: " + fs["answer"])
            parts.append("")
        parts.append("Category: " + example["category"])
        parts.append("Question: " + example["question"])
        parts.append("Answer:")
        prompt = "\n".join(parts)
        expected = " " + example["answer"]
        return prompt, expected, 0


class BigBenchQAWikiData(LanguageModelTask):
    def __init__(self):
        super().__init__(
            name="bb_qa_wikidata",
            task_type="LM",
            num_shots=10,
            category="world_knowledge",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lukaemon/bbh", "qa_wikidata", split="test")
        self.data = []
        for ex in dataset:
            self.data.append({"input": ex["input"], "target": ex["target"]})

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            parts.append("Q: " + fs["input"])
            parts.append("A: " + fs["target"])
            parts.append("")
        parts.append("Q: " + example["input"])
        parts.append("A:")
        prompt = "\n".join(parts)
        expected = " " + example["target"]
        return prompt, expected, 0


class ARCEasy(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="arc_easy",
            task_type="MC",
            num_shots=10,
            category="world_knowledge",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("allenai/arc", "ARC-Easy", split="validation")
        self.data = []
        for ex in dataset:
            question = ex["question"]
            choices = ex["choices"]
            text_choices = choices["text"]
            label_map = choices["label"]
            answer_key = ex["answerKey"]
            label_idx = label_map.index(answer_key)
            self.data.append({
                "question": question,
                "choices": text_choices,
                "label": label_idx,
            })

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            q = fs["question"]
            choices_str = "\n".join(
                f"{chr(65 + i)}. {c}" for i, c in enumerate(fs["choices"])
            )
            parts.append("Question: " + q + "\n" + choices_str)
            ans_letter = chr(65 + fs["label"])
            parts.append("Answer: " + ans_letter)
            parts.append("")
        choices_str = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(example["choices"])
        )
        parts.append("Question: " + example["question"] + "\n" + choices_str)
        parts.append("Answer:")
        prompt = "\n".join(parts)
        choices = [chr(65 + i) for i in range(len(example["choices"]))]
        return prompt, choices, example["label"]


class ARCChallenge(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="arc_challenge",
            task_type="MC",
            num_shots=10,
            category="world_knowledge",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("allenai/arc", "ARC-Challenge", split="validation")
        self.data = []
        for ex in dataset:
            question = ex["question"]
            choices = ex["choices"]
            text_choices = choices["text"]
            label_map = choices["label"]
            answer_key = ex["answerKey"]
            label_idx = label_map.index(answer_key)
            self.data.append({
                "question": question,
                "choices": text_choices,
                "label": label_idx,
            })

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            q = fs["question"]
            choices_str = "\n".join(
                f"{chr(65 + i)}. {c}" for i, c in enumerate(fs["choices"])
            )
            parts.append("Question: " + q + "\n" + choices_str)
            ans_letter = chr(65 + fs["label"])
            parts.append("Answer: " + ans_letter)
            parts.append("")
        choices_str = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(example["choices"])
        )
        parts.append("Question: " + example["question"] + "\n" + choices_str)
        parts.append("Answer:")
        prompt = "\n".join(parts)
        choices = [chr(65 + i) for i in range(len(example["choices"]))]
        return prompt, choices, example["label"]


class MMLU0Shot(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="mmlu_0shot",
            task_type="MC",
            num_shots=0,
            category="world_knowledge",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("cais/mmlu", "all", split="test")
        self.data = []
        for ex in dataset:
            question = ex["question"]
            choices = ex["choices"]
            answer_key = ex["answer"]
            label_idx = int(answer_key)
            self.data.append({
                "question": question,
                "choices": choices,
                "label": label_idx,
                "subject": ex.get("subject", ""),
            })

    def format_prompt(self, example, few_shot_examples):
        parts = []
        if example["subject"]:
            parts.append(f"The following are multiple choice questions about {example['subject']}.")
        choices_str = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(example["choices"])
        )
        parts.append("Question: " + example["question"] + "\n" + choices_str)
        parts.append("Answer:")
        prompt = "\n".join(parts)
        choices = [chr(65 + i) for i in range(len(example["choices"]))]
        return prompt, choices, example["label"]


class MMLU5Shot(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="mmlu_5shot",
            task_type="MC",
            num_shots=5,
            category="world_knowledge",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("cais/mmlu", "all", split="test")
        self.data = []
        for ex in dataset:
            question = ex["question"]
            choices = ex["choices"]
            answer_key = ex["answer"]
            label_idx = int(answer_key)
            self.data.append({
                "question": question,
                "choices": choices,
                "label": label_idx,
                "subject": ex.get("subject", ""),
            })
        random.Random(42).shuffle(self.data)

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            choices_str = "\n".join(
                f"{chr(65 + i)}. {c}" for i, c in enumerate(fs["choices"])
            )
            parts.append("Question: " + fs["question"] + "\n" + choices_str)
            ans_letter = chr(65 + fs["label"])
            parts.append("Answer: " + ans_letter)
            parts.append("")
        choices_str = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(example["choices"])
        )
        parts.append("Question: " + example["question"] + "\n" + choices_str)
        parts.append("Answer:")
        prompt = "\n".join(parts)
        choices = [chr(65 + i) for i in range(len(example["choices"]))]
        return prompt, choices, example["label"]


class BigBenchMisconceptions(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="bb_misconceptions",
            task_type="MC",
            num_shots=10,
            category="world_knowledge",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lukaemon/bbh", "misconceptions", split="test")
        self.data = []
        for ex in dataset:
            self.data.append({"input": ex["input"], "target": ex["target"]})

    def format_prompt(self, example, few_shot_examples):
        input_text = example["input"]
        target = example["target"]
        choices = ["True", "False"]
        label = 0
        if "false" in target.lower() or "no" in target.lower():
            label = 1
        parts = []
        for fs in few_shot_examples:
            parts.append("Q: " + fs["input"])
            fs_label = 1 if ("false" in fs["target"].lower() or "no" in fs["target"].lower()) else 0
            parts.append("A: " + choices[fs_label])
            parts.append("")
        parts.append("Q: " + input_text)
        parts.append("A:")
        prompt = "\n".join(parts)
        return prompt, choices, label
