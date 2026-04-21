import random
from datasets import load_dataset
from .base import MultipleChoiceTask, LanguageModelTask, SchemaTask


class COPA(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="copa",
            task_type="MC",
            num_shots=0,
            category="commonsense",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("super_glue", "copa", split="validation")
        self.data = []
        for ex in dataset:
            premise = ex["premise"]
            question = ex["question"]
            choice1 = ex["choice1"]
            choice2 = ex["choice2"]
            label = ex["label"]
            self.data.append({
                "premise": premise,
                "question": question,
                "choice1": choice1,
                "choice2": choice2,
                "label": label,
            })

    def format_prompt(self, example, few_shot_examples):
        premise = example["premise"]
        question = example["question"]
        choice1 = example["choice1"]
        choice2 = example["choice2"]
        label = example["label"]
        prompt = premise + "\n" + question + "\n"
        choices = [choice1, choice2]
        return prompt, choices, label


class CommonsenseQA(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="commonsenseqa",
            task_type="MC",
            num_shots=10,
            category="commonsense",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("tau/commonsense_qa", split="validation")
        self.data = []
        for ex in dataset:
            question = ex["question"]
            choices_text = ex["choices"]["text"]
            choices_label = ex["choices"]["label"]
            answer_key = ex["answerKey"]
            label_idx = choices_label.index(answer_key)
            self.data.append({
                "question": question,
                "choices": choices_text,
                "label": label_idx,
                "answer_key": answer_key,
            })

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


class PIQA(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="piqa",
            task_type="MC",
            num_shots=10,
            category="commonsense",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("piqa", split="validation")
        self.data = []
        for ex in dataset:
            goal = ex["goal"]
            sol1 = ex["sol1"]
            sol2 = ex["sol2"]
            label = ex["label"]
            self.data.append({
                "goal": goal,
                "sol1": sol1,
                "sol2": sol2,
                "label": label,
            })

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            parts.append("Goal: " + fs["goal"])
            ans_choice = fs["sol1"] if fs["label"] == 0 else fs["sol2"]
            parts.append("Solution: " + ans_choice)
            parts.append("")
        parts.append("Goal: " + example["goal"])
        parts.append("Solution:")
        prompt = "\n".join(parts)
        choices = [example["sol1"], example["sol2"]]
        return prompt, choices, example["label"]


class OpenBookQA(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="openbookqa",
            task_type="MC",
            num_shots=0,
            category="commonsense",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("openbookqa", "main", split="test")
        self.data = []
        for ex in dataset:
            question = ex["question_stem"]
            choices_text = ex["choices"]["text"]
            choices_label = ex["choices"]["label"]
            answer_key = ex["answerKey"]
            label_idx = choices_label.index(answer_key)
            self.data.append({
                "question": question,
                "choices": choices_text,
                "label": label_idx,
            })

    def format_prompt(self, example, few_shot_examples):
        question = example["question"]
        choices = example["choices"]
        label = example["label"]
        choices_str = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
        )
        prompt = "Question: " + question + "\n" + choices_str + "\nAnswer:"
        choice_labels = [chr(65 + i) for i in range(len(choices))]
        return prompt, choice_labels, label


class SIQA(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="siqa",
            task_type="MC",
            num_shots=10,
            category="commonsense",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("social_i_qa", split="validation")
        self.data = []
        for ex in dataset:
            context = ex["context"]
            question = ex["question"]
            answer_a = ex["answerA"]
            answer_b = ex["answerB"]
            answer_c = ex["answerC"]
            label_str = ex.get("label", "1")
            label_idx = int(label_str) - 1
            if label_idx < 0 or label_idx > 2:
                label_idx = 0
            self.data.append({
                "context": context,
                "question": question,
                "answerA": answer_a,
                "answerB": answer_b,
                "answerC": answer_c,
                "label": label_idx,
            })

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            parts.append("Context: " + fs["context"])
            parts.append("Question: " + fs["question"])
            ans_choices = [fs["answerA"], fs["answerB"], fs["answerC"]]
            parts.append("Answer: " + ans_choices[fs["label"]])
            parts.append("")
        parts.append("Context: " + example["context"])
        parts.append("Question: " + example["question"])
        parts.append("Answer:")
        prompt = "\n".join(parts)
        choices = [example["answerA"], example["answerB"], example["answerC"]]
        return prompt, choices, example["label"]


class BigBenchNovelConcepts(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="bb_novel_concepts",
            task_type="MC",
            num_shots=10,
            category="commonsense",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lukaemon/bbh", "novel_concepts", split="test")
        self.data = []
        for ex in dataset:
            input_text = ex["input"]
            target = ex["target"]
            choices = self._extract_choices(input_text)
            if not choices:
                choices = target.split(", ")
            self.data.append({"input": input_text, "target": target, "choices": choices})

    def _extract_choices(self, text):
        choices = []
        if "(A)" in text:
            parts = text.split("(A)")
            if len(parts) > 1:
                rest = parts[1]
                for letter in ["A", "B", "C", "D", "E"]:
                    pattern = f"({letter})"
                    if pattern in rest:
                        idx = rest.index(pattern)
                        rest = rest[idx + len(pattern):]
        for marker in ["(A)", "(B)", "(C)", "(D)"]:
            idx = text.find(marker)
            if idx >= 0:
                start = idx + len(marker)
                end = text.find("(", start + 1) if start + 1 < len(text) else len(text)
                if end < 0:
                    end = len(text)
                choice = text[start:end].strip()
                if choice:
                    choices.append(choice)
        return choices

    def format_prompt(self, example, few_shot_examples):
        target = example["target"]
        choices = example["choices"]
        label = 0
        for i, c in enumerate(choices):
            if c.strip().lower() == target.strip().lower():
                label = i
                break
        parts = []
        for fs in few_shot_examples:
            parts.append("Q: " + fs["input"])
            parts.append("A: " + fs["target"])
            parts.append("")
        parts.append("Q: " + example["input"])
        parts.append("A:")
        prompt = "\n".join(parts)
        return prompt, choices, label


class BigBenchStrangeStories(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="bb_strange_stories",
            task_type="MC",
            num_shots=10,
            category="commonsense",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lukaemon/bbh", "strange_stories", split="test")
        self.data = []
        for ex in dataset:
            self.data.append({"input": ex["input"], "target": ex["target"]})

    def format_prompt(self, example, few_shot_examples):
        target = example["target"]
        choices = ["Yes", "No"]
        label = 0
        if "no" in target.lower():
            label = 1
        parts = []
        for fs in few_shot_examples:
            parts.append("Q: " + fs["input"])
            fs_label = 1 if "no" in fs["target"].lower() else 0
            parts.append("A: " + choices[fs_label])
            parts.append("")
        parts.append("Q: " + example["input"])
        parts.append("A:")
        prompt = "\n".join(parts)
        return prompt, choices, label


class BigBenchStrategyQA(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="bb_strategy_qa",
            task_type="MC",
            num_shots=10,
            category="commonsense",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lukaemon/bbh", "strategyqa", split="test")
        self.data = []
        for ex in dataset:
            self.data.append({"input": ex["input"], "target": ex["target"]})

    def format_prompt(self, example, few_shot_examples):
        target = example["target"]
        choices = ["yes", "no"]
        label = 0
        if "no" in target.lower():
            label = 1
        parts = []
        for fs in few_shot_examples:
            parts.append("Q: " + fs["input"])
            fs_label = 1 if "no" in fs["target"].lower() else 0
            parts.append("A: " + choices[fs_label])
            parts.append("")
        parts.append("Q: " + example["input"])
        parts.append("A:")
        prompt = "\n".join(parts)
        return prompt, choices, label
