import random
from datasets import load_dataset
from .base import MultipleChoiceTask, LanguageModelTask, SchemaTask


class HellaSwag0Shot(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="hellaswag_0shot",
            task_type="MC",
            num_shots=0,
            category="understanding",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("hellaswag", split="validation")
        self.data = []
        for ex in dataset:
            ctx = ex["ctx"]
            endings = [ex["endings"][0], ex["endings"][1], ex["endings"][2], ex["endings"][3]]
            label = ex["label"]
            self.data.append({"context": ctx, "endings": endings, "label": label})

    def format_prompt(self, example, few_shot_examples):
        prompt = example["context"]
        choices = example["endings"]
        label = example["label"]
        return prompt, choices, label


class HellaSwag10Shot(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="hellaswag_10shot",
            task_type="MC",
            num_shots=10,
            category="understanding",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("hellaswag", split="validation")
        self.data = []
        for ex in dataset:
            ctx = ex["ctx"]
            endings = [ex["endings"][0], ex["endings"][1], ex["endings"][2], ex["endings"][3]]
            label = ex["label"]
            self.data.append({"context": ctx, "endings": endings, "label": label})

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            parts.append(fs["context"])
            parts.append(fs["endings"][fs["label"]])
        parts.append(example["context"])
        prompt = "\n".join(parts)
        choices = example["endings"]
        label = example["label"]
        return prompt, choices, label


class LambadaTask(LanguageModelTask):
    def __init__(self):
        super().__init__(
            name="lambada",
            task_type="LM",
            num_shots=0,
            category="understanding",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lambada", split="validation")
        self.data = []
        for ex in dataset:
            text = ex["text"].strip()
            words = text.split()
            if len(words) < 2:
                continue
            context = " ".join(words[:-1])
            last_word = words[-1]
            self.data.append({"context": context, "expected": last_word, "label": 0})

    def format_prompt(self, example, few_shot_examples):
        prompt = example["context"]
        expected = " " + example["expected"]
        return prompt, expected, example["label"]


class WinogradWSC(SchemaTask):
    def __init__(self):
        super().__init__(
            name="winograd_wsc",
            task_type="S",
            num_shots=0,
            category="understanding",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("winograd_wsc", "wsc273", split="test")
        self.data = []
        for ex in dataset:
            text = ex["text"]
            pronoun = ex.get("pronoun", ex.get("pron", ""))
            pronoun_loc = ex.get("pronoun_loc", ex.get("pron_loc", -1))
            candidates = ex.get("candidates", ex.get("options", ["the man", "the woman"]))
            if isinstance(candidates, str):
                candidates = [candidates]
            label_idx = ex.get("label", 0)
            if isinstance(label_idx, str):
                label_idx = int(label_idx)
            self.data.append({
                "text": text,
                "pronoun": pronoun,
                "candidates": candidates,
                "label": label_idx,
            })

    def format_prompt(self, example, few_shot_examples):
        text = example["text"]
        pronoun = example["pronoun"]
        candidates = example["candidates"]
        label = example["label"]
        prompt = text + "\nWhat does \"" + pronoun + "\" refer to?"

        def schema_fn():
            options = []
            for i, c in enumerate(candidates):
                completion = c.strip()
                options.append((i, completion))
            return options

        return prompt, schema_fn, label


class WinoGrande(SchemaTask):
    def __init__(self):
        super().__init__(
            name="winogrande",
            task_type="S",
            num_shots=0,
            category="understanding",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("winogrande", "winogrande_xl", split="validation")
        self.data = []
        for ex in dataset:
            sentence = ex["sentence"]
            option1 = ex["option1"]
            option2 = ex["option2"]
            answer = ex["answer"]
            label_idx = int(answer) - 1
            self.data.append({
                "sentence": sentence,
                "option1": option1,
                "option2": option2,
                "label": label_idx,
            })

    def format_prompt(self, example, few_shot_examples):
        sentence = example["sentence"]
        option1 = example["option1"]
        option2 = example["option2"]
        label = example["label"]
        prompt = sentence + "\nChoose the correct option."

        def schema_fn():
            return [(0, option1), (1, option2)]

        return prompt, schema_fn, label


class BigBenchLanguageID(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="bb_language_id",
            task_type="MC",
            num_shots=10,
            category="understanding",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lukaemon/bbh", "language_identification", split="test")
        self.data = []
        for ex in dataset:
            self.data.append({"input": ex["input"], "target": ex["target"]})

    def format_prompt(self, example, few_shot_examples):
        choices = ["English", "French", "German", "Spanish", "Italian", "Chinese", "Japanese", "Korean", "Russian", "Portuguese", "Arabic", "Hindi", "Dutch", "Turkish", "Polish"]
        label = 0
        target = example["target"]
        for i, c in enumerate(choices):
            if c.lower() in target.lower():
                label = i
                break
        parts = []
        for fs in few_shot_examples:
            parts.append("Input: " + fs["input"])
            parts.append("Output: " + fs["target"])
        parts.append("Input: " + example["input"])
        parts.append("Output:")
        prompt = "\n".join(parts)
        return prompt, choices, label


class BigBenchConlangTranslation(LanguageModelTask):
    def __init__(self):
        super().__init__(
            name="bb_conlang_translation",
            task_type="LM",
            num_shots=0,
            category="understanding",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lukaemon/bbh", "conlang_translation", split="test")
        self.data = []
        for ex in dataset:
            self.data.append({"input": ex["input"], "target": ex["target"]})

    def format_prompt(self, example, few_shot_examples):
        prompt = example["input"] + "\n"
        expected = example["target"]
        return prompt, expected, 0


class BigBenchConceptualCombinations(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="bb_conceptual_combinations",
            task_type="MC",
            num_shots=10,
            category="understanding",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lukaemon/bbh", "conceptual_combinations", split="test")
        self.data = []
        for ex in dataset:
            input_text = ex["input"]
            target = ex["target"]
            choices = self._extract_choices(input_text)
            if not choices:
                target_words = target.split(",")
                choices = [w.strip() for w in target_words]
            self.data.append({"input": input_text, "target": target, "choices": choices})

    def _extract_choices(self, text):
        choices = []
        parts = text.split("Options:")
        if len(parts) > 1:
            option_part = parts[1].strip()
            for line in option_part.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-") or line.startswith("(")):
                    cleaned = line.lstrip("0123456789.-)(").strip()
                    if cleaned:
                        choices.append(cleaned)
        return choices

    def format_prompt(self, example, few_shot_examples):
        input_text = example["input"]
        target = example["target"]
        choices = example["choices"]
        if not choices:
            choices = [w.strip() for w in target.split(",")]
        label = 0
        for i, c in enumerate(choices):
            if c.strip().lower() == target.strip().lower():
                label = i
                break
        parts = []
        for fs in few_shot_examples:
            parts.append("Q: " + fs["input"])
            parts.append("A: " + fs["target"])
        parts.append("Q: " + input_text)
        parts.append("A:")
        prompt = "\n".join(parts)
        return prompt, choices, label
