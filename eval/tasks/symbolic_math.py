import random
import string
from datasets import load_dataset
from .base import MultipleChoiceTask, LanguageModelTask, SchemaTask


class BigBenchDyckLanguages(LanguageModelTask):
    def __init__(self):
        super().__init__(
            name="bb_dyck_languages",
            task_type="LM",
            num_shots=10,
            category="symbolic_math",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lukaemon/bbh", "dyck_languages", split="test")
        self.data = []
        for ex in dataset:
            self.data.append({"input": ex["input"], "target": ex["target"]})

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            parts.append("Input: " + fs["input"])
            parts.append("Output: " + fs["target"])
            parts.append("")
        parts.append("Input: " + example["input"])
        parts.append("Output:")
        prompt = "\n".join(parts)
        expected = " " + example["target"]
        return prompt, expected, 0


class AGIEvalLSATAR(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="agi_eval_lsat_ar",
            task_type="MC",
            num_shots=3,
            category="symbolic_math",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("m-a-p/AGIEval", "lsat-ar", split="test")
        self.data = []
        for ex in dataset:
            passage = ex.get("passage", "")
            question = ex.get("question", "")
            if passage:
                full_question = passage + "\n" + question
            else:
                full_question = question
            options = ex.get("options", [])
            label_str = ex.get("label", "A")
            label_idx = ord(label_str.strip()) - ord("A") if isinstance(label_str, str) and len(label_str.strip()) == 1 else int(label_str)
            if isinstance(options, str):
                options = [opt.strip() for opt in options.split("\n") if opt.strip()]
            self.data.append({
                "question": full_question,
                "options": options,
                "label": label_idx,
            })

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            choices_str = "\n".join(
                f"{chr(65 + i)}. {c}" for i, c in enumerate(fs["options"])
            )
            parts.append("Question: " + fs["question"] + "\n" + choices_str)
            ans_letter = chr(65 + fs["label"])
            parts.append("Answer: " + ans_letter)
            parts.append("")
        choices_str = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(example["options"])
        )
        parts.append("Question: " + example["question"] + "\n" + choices_str)
        parts.append("Answer:")
        prompt = "\n".join(parts)
        choices = [chr(65 + i) for i in range(len(example["options"]))]
        return prompt, choices, example["label"]


class BigBenchCSAlgorithms(LanguageModelTask):
    def __init__(self):
        super().__init__(
            name="bb_cs_algorithms",
            task_type="LM",
            num_shots=10,
            category="symbolic_math",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lukaemon/bbh", "cs_algorithms", split="test")
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


class BigBenchOperators(LanguageModelTask):
    def __init__(self):
        super().__init__(
            name="bb_operators",
            task_type="LM",
            num_shots=10,
            category="symbolic_math",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lukaemon/bbh", "operators", split="test")
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


class BigBenchRepeatCopyLogic(LanguageModelTask):
    def __init__(self):
        super().__init__(
            name="bb_repeat_copy_logic",
            task_type="LM",
            num_shots=10,
            category="symbolic_math",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lukaemon/bbh", "repeat_copy_logic", split="test")
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


class BigBenchElementaryMathQA(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="bb_elementary_math_qa",
            task_type="MC",
            num_shots=10,
            category="symbolic_math",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lukaemon/bbh", "elementary_math_qa", split="test")
        self.data = []
        for ex in dataset:
            self.data.append({"input": ex["input"], "target": ex["target"]})

    def format_prompt(self, example, few_shot_examples):
        target = example["target"]
        choices = ["A", "B", "C", "D", "E"]
        label = 0
        for i, c in enumerate(choices):
            if c == target.strip():
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


class BigBenchLogicalDeduction(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="bb_logical_deduction",
            task_type="MC",
            num_shots=10,
            category="symbolic_math",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lukaemon/bbh", "logical_deduction", split="test")
        self.data = []
        for ex in dataset:
            self.data.append({"input": ex["input"], "target": ex["target"]})

    def format_prompt(self, example, few_shot_examples):
        target = example["target"]
        choices = ["A", "B", "C"]
        label = 0
        for i, c in enumerate(choices):
            if c == target.strip():
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


class SimpleArithmeticNoSpaces(LanguageModelTask):
    def __init__(self):
        super().__init__(
            name="simple_arithmetic_no_spaces",
            task_type="LM",
            num_shots=10,
            category="symbolic_math",
            is_core=False,
            is_core_ext=True,
        )
        self.examples = self._generate_examples(500)

    def _generate_examples(self, n):
        examples = []
        rng = random.Random(42)
        for _ in range(n):
            a = rng.randint(0, 999)
            b = rng.randint(0, 999)
            op = rng.choice(["+", "-", "*"])
            if op == "+":
                result = a + b
            elif op == "-":
                result = a - b
            else:
                a = rng.randint(0, 99)
                b = rng.randint(0, 99)
                result = a * b
            expr = f"{a}{op}{b}"
            answer = str(result)
            examples.append({"input": expr, "target": answer})
        return examples

    def load_data(self):
        self.data = self.examples

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            parts.append(fs["input"] + "=" + fs["target"])
        parts.append(example["input"] + "=")
        prompt = ", ".join(parts)
        expected = example["target"]
        return prompt, expected, 0


class SimpleArithmeticWithSpaces(LanguageModelTask):
    def __init__(self):
        super().__init__(
            name="simple_arithmetic_with_spaces",
            task_type="LM",
            num_shots=10,
            category="symbolic_math",
            is_core=False,
            is_core_ext=True,
        )
        self.examples = self._generate_examples(500)

    def _generate_examples(self, n):
        examples = []
        rng = random.Random(42)
        for _ in range(n):
            a = rng.randint(0, 999)
            b = rng.randint(0, 999)
            op = rng.choice(["+", "-", "*"])
            if op == "+":
                result = a + b
            elif op == "-":
                result = a - b
            else:
                a = rng.randint(0, 99)
                b = rng.randint(0, 99)
                result = a * b
            expr = f"{a} {op} {b}"
            answer = str(result)
            examples.append({"input": expr, "target": answer})
        return examples

    def load_data(self):
        self.data = self.examples

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            parts.append(fs["input"] + " = " + fs["target"])
        parts.append(example["input"] + " = ")
        prompt = "\n".join(parts)
        expected = " " + example["target"]
        return prompt, expected, 0


class MathQA(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="mathqa",
            task_type="MC",
            num_shots=10,
            category="symbolic_math",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("math_qa", split="test")
        self.data = []
        for ex in dataset:
            problem = ex["Problem"]
            options = ex["options"]
            answer = ex["correct"]
            label_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
            label_idx = label_map.get(answer.lower(), 0)
            self.data.append({
                "problem": problem,
                "options": options,
                "label": label_idx,
            })

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            parts.append("Problem: " + fs["problem"])
            parts.append("Options: " + fs["options"])
            ans_letter = chr(65 + fs["label"])
            parts.append("Answer: " + ans_letter)
            parts.append("")
        parts.append("Problem: " + example["problem"])
        parts.append("Options: " + example["options"])
        parts.append("Answer:")
        prompt = "\n".join(parts)
        choices = [chr(65 + i) for i in range(5)]
        return prompt, choices, example["label"]


class LogiQA(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="logiqa",
            task_type="MC",
            num_shots=10,
            category="symbolic_math",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lucadiliello/logiqa", split="test")
        self.data = []
        for ex in dataset:
            context = ex.get("context", "")
            question = ex.get("question", "")
            options = ex.get("options", [])
            label = ex.get("label", 0)
            if isinstance(label, str):
                try:
                    label = int(label)
                except ValueError:
                    label = 0
            if isinstance(options, str):
                options = [opt.strip() for opt in options.split("\n") if opt.strip()]
            self.data.append({
                "context": context,
                "question": question,
                "options": options,
                "label": label,
            })

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            parts.append("Passage: " + fs["context"])
            parts.append("Question: " + fs["question"])
            choices_str = "\n".join(
                f"{chr(65 + i)}. {c}" for i, c in enumerate(fs["options"])
            )
            parts.append(choices_str)
            ans_letter = chr(65 + fs["label"])
            parts.append("Answer: " + ans_letter)
            parts.append("")
        parts.append("Passage: " + example["context"])
        parts.append("Question: " + example["question"])
        choices_str = "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(example["options"])
        )
        parts.append(choices_str)
        parts.append("Answer:")
        prompt = "\n".join(parts)
        choices = [chr(65 + i) for i in range(len(example["options"]))]
        return prompt, choices, example["label"]
