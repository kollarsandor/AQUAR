import random
from datasets import load_dataset
from .base import MultipleChoiceTask, LanguageModelTask, SchemaTask


class SQuADTask(LanguageModelTask):
    def __init__(self):
        super().__init__(
            name="squad",
            task_type="LM",
            num_shots=10,
            category="reading_comp",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("squad", split="validation")
        self.data = []
        for ex in dataset:
            context = ex["context"]
            question = ex["question"]
            answers = ex["answers"]["text"]
            if not answers:
                continue
            answer = answers[0]
            self.data.append({
                "context": context,
                "question": question,
                "answer": answer,
            })

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            parts.append("Context: " + fs["context"])
            parts.append("Question: " + fs["question"])
            parts.append("Answer: " + fs["answer"])
            parts.append("")
        parts.append("Context: " + example["context"])
        parts.append("Question: " + example["question"])
        parts.append("Answer:")
        prompt = "\n".join(parts)
        expected = " " + example["answer"]
        return prompt, expected, 0


class CoQATask(LanguageModelTask):
    def __init__(self):
        super().__init__(
            name="coqa",
            task_type="LM",
            num_shots=0,
            category="reading_comp",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("coqa", split="validation")
        self.data = []
        for ex in dataset:
            story = ex["story"]
            questions = ex["questions"]
            answers = ex["answers"]
            for i in range(min(len(questions), 3)):
                q_text = questions[i] if isinstance(questions[i], str) else questions[i].get("input_text", str(questions[i]))
                a_text = answers[i] if isinstance(answers[i], str) else answers[i].get("input_text", str(answers[i]))
                self.data.append({
                    "story": story,
                    "question": q_text,
                    "answer": a_text,
                })

    def format_prompt(self, example, few_shot_examples):
        prompt = "Story: " + example["story"] + "\nQuestion: " + example["question"] + "\nAnswer:"
        expected = " " + example["answer"]
        return prompt, expected, 0


class BoolQ(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="boolq",
            task_type="MC",
            num_shots=10,
            category="reading_comp",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("super_glue", "boolq", split="validation")
        self.data = []
        for ex in dataset:
            passage = ex["passage"]
            question = ex["question"]
            label = ex["label"]
            self.data.append({
                "passage": passage,
                "question": question,
                "label": label,
            })

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            parts.append("Passage: " + fs["passage"])
            parts.append("Question: " + fs["question"] + " True or False?")
            ans = "True" if fs["label"] == 1 else "False"
            parts.append("Answer: " + ans)
            parts.append("")
        parts.append("Passage: " + example["passage"])
        parts.append("Question: " + example["question"] + " True or False?")
        parts.append("Answer:")
        prompt = "\n".join(parts)
        choices = ["True", "False"]
        return prompt, choices, example["label"]


class PubMedQALabeled(LanguageModelTask):
    def __init__(self):
        super().__init__(
            name="pubmedqa_labeled",
            task_type="LM",
            num_shots=10,
            category="reading_comp",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
        self.data = []
        for ex in dataset:
            context = ex["context"]
            if isinstance(context, dict):
                contexts = context.get("contexts", [])
                labels = context.get("labels", [])
                meshes = context.get("meshes", [])
                context_str = " ".join(contexts) if contexts else ""
            else:
                context_str = str(context)
            question = ex.get("question", "")
            answer = ex.get("final_decision", "yes")
            if not question or not context_str:
                continue
            self.data.append({
                "context": context_str,
                "question": question,
                "answer": answer,
            })

    def format_prompt(self, example, few_shot_examples):
        parts = []
        for fs in few_shot_examples:
            parts.append("Context: " + fs["context"])
            parts.append("Question: " + fs["question"])
            parts.append("Answer: " + fs["answer"])
            parts.append("")
        parts.append("Context: " + example["context"])
        parts.append("Question: " + example["question"])
        parts.append("Answer:")
        prompt = "\n".join(parts)
        expected = " " + example["answer"]
        return prompt, expected, 0


class AGIEvalLSATRC(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="agi_eval_lsat_rc",
            task_type="MC",
            num_shots=3,
            category="reading_comp",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("m-a-p/AGIEval", "lsat-rc", split="test")
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


class AGIEvalLSATLR(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="agi_eval_lsat_lr",
            task_type="MC",
            num_shots=3,
            category="reading_comp",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("m-a-p/AGIEval", "lsat-lr", split="test")
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


class AGIEvalSATEnglish(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="agi_eval_sat_english",
            task_type="MC",
            num_shots=3,
            category="reading_comp",
            is_core=False,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("m-a-p/AGIEval", "sat-en", split="test")
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


class BigBenchUnderstandingFables(MultipleChoiceTask):
    def __init__(self):
        super().__init__(
            name="bb_understanding_fables",
            task_type="MC",
            num_shots=10,
            category="reading_comp",
            is_core=True,
            is_core_ext=True,
        )

    def load_data(self):
        dataset = load_dataset("lukaemon/bbh", "ruin_names", split="test")
        self.data = []
        for ex in dataset:
            self.data.append({"input": ex["input"], "target": ex["target"]})

    def format_prompt(self, example, few_shot_examples):
        target = example["target"]
        input_text = example["input"]
        choices = ["A", "B", "C", "D"]
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
        parts.append("Q: " + input_text)
        parts.append("A:")
        prompt = "\n".join(parts)
        return prompt, choices, label
