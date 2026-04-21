from .base import BaseTask, MultipleChoiceTask, LanguageModelTask, SchemaTask
from .understanding import (
    HellaSwag0Shot,
    HellaSwag10Shot,
    LambadaTask,
    WinogradWSC,
    WinoGrande,
    BigBenchLanguageID,
    BigBenchConlangTranslation,
    BigBenchConceptualCombinations,
)
from .world_knowledge import (
    JeopardyTask,
    BigBenchQAWikiData,
    ARCEasy,
    ARCChallenge,
    MMLU0Shot,
    MMLU5Shot,
    BigBenchMisconceptions,
)
from .commonsense import (
    COPA,
    CommonsenseQA,
    PIQA,
    OpenBookQA,
    SIQA,
    BigBenchNovelConcepts,
    BigBenchStrangeStories,
    BigBenchStrategyQA,
)
from .symbolic_math import (
    BigBenchDyckLanguages,
    AGIEvalLSATAR,
    BigBenchCSAlgorithms,
    BigBenchOperators,
    BigBenchRepeatCopyLogic,
    BigBenchElementaryMathQA,
    BigBenchLogicalDeduction,
    SimpleArithmeticNoSpaces,
    SimpleArithmeticWithSpaces,
    MathQA,
    LogiQA,
)
from .reading_comp import (
    SQuADTask,
    CoQATask,
    BoolQ,
    PubMedQALabeled,
    AGIEvalLSATRC,
    AGIEvalLSATLR,
    AGIEvalSATEnglish,
    BigBenchUnderstandingFables,
)
from .safety import (
    WinogenderFemale,
    WinogenderMale,
    EnterprisePIIClassification,
    BBQTask,
)


def _make_task(name):
    mapping = {
        "hellaswag_0shot": lambda: HellaSwag0Shot(),
        "hellaswag_10shot": lambda: HellaSwag10Shot(),
        "lambada": lambda: LambadaTask(),
        "winograd_wsc": lambda: WinogradWSC(),
        "winogrande": lambda: WinoGrande(),
        "bb_language_id": lambda: BigBenchLanguageID(),
        "bb_conlang_translation": lambda: BigBenchConlangTranslation(),
        "bb_conceptual_combinations": lambda: BigBenchConceptualCombinations(),
        "jeopardy": lambda: JeopardyTask(),
        "bb_qa_wikidata": lambda: BigBenchQAWikiData(),
        "arc_easy": lambda: ARCEasy(),
        "arc_challenge": lambda: ARCChallenge(),
        "mmlu_0shot": lambda: MMLU0Shot(),
        "mmlu_5shot": lambda: MMLU5Shot(),
        "bb_misconceptions": lambda: BigBenchMisconceptions(),
        "copa": lambda: COPA(),
        "commonsenseqa": lambda: CommonsenseQA(),
        "piqa": lambda: PIQA(),
        "openbookqa": lambda: OpenBookQA(),
        "siqa": lambda: SIQA(),
        "bb_novel_concepts": lambda: BigBenchNovelConcepts(),
        "bb_strange_stories": lambda: BigBenchStrangeStories(),
        "bb_strategy_qa": lambda: BigBenchStrategyQA(),
        "bb_dyck_languages": lambda: BigBenchDyckLanguages(),
        "agi_eval_lsat_ar": lambda: AGIEvalLSATAR(),
        "bb_cs_algorithms": lambda: BigBenchCSAlgorithms(),
        "bb_operators": lambda: BigBenchOperators(),
        "bb_repeat_copy_logic": lambda: BigBenchRepeatCopyLogic(),
        "bb_elementary_math_qa": lambda: BigBenchElementaryMathQA(),
        "bb_logical_deduction": lambda: BigBenchLogicalDeduction(),
        "simple_arithmetic_no_spaces": lambda: SimpleArithmeticNoSpaces(),
        "simple_arithmetic_with_spaces": lambda: SimpleArithmeticWithSpaces(),
        "mathqa": lambda: MathQA(),
        "logiqa": lambda: LogiQA(),
        "squad": lambda: SQuADTask(),
        "coqa": lambda: CoQATask(),
        "boolq": lambda: BoolQ(),
        "pubmedqa_labeled": lambda: PubMedQALabeled(),
        "agi_eval_lsat_rc": lambda: AGIEvalLSATRC(),
        "agi_eval_lsat_lr": lambda: AGIEvalLSATLR(),
        "agi_eval_sat_english": lambda: AGIEvalSATEnglish(),
        "bb_understanding_fables": lambda: BigBenchUnderstandingFables(),
        "winogender_female": lambda: WinogenderFemale(),
        "winogender_male": lambda: WinogenderMale(),
        "enterprise_pii": lambda: EnterprisePIIClassification(),
        "bbq": lambda: BBQTask(),
    }
    if name not in mapping:
        raise ValueError(f"Unknown task: {name}")
    return mapping[name]()


TASK_REGISTRY = {
    "hellaswag_0shot": {
        "class": HellaSwag0Shot,
        "category": "understanding",
        "task_type": "MC",
        "is_core": True,
        "is_core_ext": True,
    },
    "hellaswag_10shot": {
        "class": HellaSwag10Shot,
        "category": "understanding",
        "task_type": "MC",
        "is_core": True,
        "is_core_ext": True,
    },
    "lambada": {
        "class": LambadaTask,
        "category": "understanding",
        "task_type": "LM",
        "is_core": True,
        "is_core_ext": True,
    },
    "winograd_wsc": {
        "class": WinogradWSC,
        "category": "understanding",
        "task_type": "S",
        "is_core": True,
        "is_core_ext": True,
    },
    "winogrande": {
        "class": WinoGrande,
        "category": "understanding",
        "task_type": "S",
        "is_core": True,
        "is_core_ext": True,
    },
    "bb_language_id": {
        "class": BigBenchLanguageID,
        "category": "understanding",
        "task_type": "MC",
        "is_core": True,
        "is_core_ext": True,
    },
    "bb_conlang_translation": {
        "class": BigBenchConlangTranslation,
        "category": "understanding",
        "task_type": "LM",
        "is_core": False,
        "is_core_ext": True,
    },
    "bb_conceptual_combinations": {
        "class": BigBenchConceptualCombinations,
        "category": "understanding",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "jeopardy": {
        "class": JeopardyTask,
        "category": "world_knowledge",
        "task_type": "LM",
        "is_core": True,
        "is_core_ext": True,
    },
    "bb_qa_wikidata": {
        "class": BigBenchQAWikiData,
        "category": "world_knowledge",
        "task_type": "LM",
        "is_core": True,
        "is_core_ext": True,
    },
    "arc_easy": {
        "class": ARCEasy,
        "category": "world_knowledge",
        "task_type": "MC",
        "is_core": True,
        "is_core_ext": True,
    },
    "arc_challenge": {
        "class": ARCChallenge,
        "category": "world_knowledge",
        "task_type": "MC",
        "is_core": True,
        "is_core_ext": True,
    },
    "mmlu_0shot": {
        "class": MMLU0Shot,
        "category": "world_knowledge",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "mmlu_5shot": {
        "class": MMLU5Shot,
        "category": "world_knowledge",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "bb_misconceptions": {
        "class": BigBenchMisconceptions,
        "category": "world_knowledge",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "copa": {
        "class": COPA,
        "category": "commonsense",
        "task_type": "MC",
        "is_core": True,
        "is_core_ext": True,
    },
    "commonsenseqa": {
        "class": CommonsenseQA,
        "category": "commonsense",
        "task_type": "MC",
        "is_core": True,
        "is_core_ext": True,
    },
    "piqa": {
        "class": PIQA,
        "category": "commonsense",
        "task_type": "MC",
        "is_core": True,
        "is_core_ext": True,
    },
    "openbookqa": {
        "class": OpenBookQA,
        "category": "commonsense",
        "task_type": "MC",
        "is_core": True,
        "is_core_ext": True,
    },
    "siqa": {
        "class": SIQA,
        "category": "commonsense",
        "task_type": "MC",
        "is_core": True,
        "is_core_ext": True,
    },
    "bb_novel_concepts": {
        "class": BigBenchNovelConcepts,
        "category": "commonsense",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "bb_strange_stories": {
        "class": BigBenchStrangeStories,
        "category": "commonsense",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "bb_strategy_qa": {
        "class": BigBenchStrategyQA,
        "category": "commonsense",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "bb_dyck_languages": {
        "class": BigBenchDyckLanguages,
        "category": "symbolic_math",
        "task_type": "LM",
        "is_core": True,
        "is_core_ext": True,
    },
    "agi_eval_lsat_ar": {
        "class": AGIEvalLSATAR,
        "category": "symbolic_math",
        "task_type": "MC",
        "is_core": True,
        "is_core_ext": True,
    },
    "bb_cs_algorithms": {
        "class": BigBenchCSAlgorithms,
        "category": "symbolic_math",
        "task_type": "LM",
        "is_core": True,
        "is_core_ext": True,
    },
    "bb_operators": {
        "class": BigBenchOperators,
        "category": "symbolic_math",
        "task_type": "LM",
        "is_core": True,
        "is_core_ext": True,
    },
    "bb_repeat_copy_logic": {
        "class": BigBenchRepeatCopyLogic,
        "category": "symbolic_math",
        "task_type": "LM",
        "is_core": True,
        "is_core_ext": True,
    },
    "bb_elementary_math_qa": {
        "class": BigBenchElementaryMathQA,
        "category": "symbolic_math",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "bb_logical_deduction": {
        "class": BigBenchLogicalDeduction,
        "category": "symbolic_math",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "simple_arithmetic_no_spaces": {
        "class": SimpleArithmeticNoSpaces,
        "category": "symbolic_math",
        "task_type": "LM",
        "is_core": False,
        "is_core_ext": True,
    },
    "simple_arithmetic_with_spaces": {
        "class": SimpleArithmeticWithSpaces,
        "category": "symbolic_math",
        "task_type": "LM",
        "is_core": False,
        "is_core_ext": True,
    },
    "mathqa": {
        "class": MathQA,
        "category": "symbolic_math",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "logiqa": {
        "class": LogiQA,
        "category": "symbolic_math",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "squad": {
        "class": SQuADTask,
        "category": "reading_comp",
        "task_type": "LM",
        "is_core": True,
        "is_core_ext": True,
    },
    "coqa": {
        "class": CoQATask,
        "category": "reading_comp",
        "task_type": "LM",
        "is_core": True,
        "is_core_ext": True,
    },
    "boolq": {
        "class": BoolQ,
        "category": "reading_comp",
        "task_type": "MC",
        "is_core": True,
        "is_core_ext": True,
    },
    "pubmedqa_labeled": {
        "class": PubMedQALabeled,
        "category": "reading_comp",
        "task_type": "LM",
        "is_core": False,
        "is_core_ext": True,
    },
    "agi_eval_lsat_rc": {
        "class": AGIEvalLSATRC,
        "category": "reading_comp",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "agi_eval_lsat_lr": {
        "class": AGIEvalLSATLR,
        "category": "reading_comp",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "agi_eval_sat_english": {
        "class": AGIEvalSATEnglish,
        "category": "reading_comp",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "bb_understanding_fables": {
        "class": BigBenchUnderstandingFables,
        "category": "reading_comp",
        "task_type": "MC",
        "is_core": True,
        "is_core_ext": True,
    },
    "winogender_female": {
        "class": WinogenderFemale,
        "category": "safety",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "winogender_male": {
        "class": WinogenderMale,
        "category": "safety",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "enterprise_pii": {
        "class": EnterprisePIIClassification,
        "category": "safety",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
    "bbq": {
        "class": BBQTask,
        "category": "safety",
        "task_type": "MC",
        "is_core": False,
        "is_core_ext": True,
    },
}

CORE_TASKS = [
    name for name, info in TASK_REGISTRY.items() if info["is_core"]
]

CORE_EXT_TASKS = [
    name for name, info in TASK_REGISTRY.items() if info["is_core_ext"]
]

CATEGORIES = [
    "understanding",
    "world_knowledge",
    "commonsense",
    "symbolic_math",
    "reading_comp",
    "safety",
]


def get_all_tasks():
    return {name: _make_task(name) for name in TASK_REGISTRY}


def get_core_tasks():
    return {name: _make_task(name) for name in CORE_TASKS}


def get_core_ext_tasks():
    return {name: _make_task(name) for name in CORE_EXT_TASKS}


def run_evaluation(model, tokenizer, device, max_recurrence=1, task_subset=None):
    if task_subset is None:
        task_subset = list(TASK_REGISTRY.keys())
    results = {}
    for task_name in task_subset:
        if task_name not in TASK_REGISTRY:
            continue
        task_info = TASK_REGISTRY[task_name]
        task = _make_task(task_name)
        task.load_data()
        seed_scores = []
        for seed in range(3):
            score = task.evaluate(model, tokenizer, device, max_recurrence=max_recurrence, seed=seed)
            seed_scores.append(score)
        avg_score = sum(seed_scores) / len(seed_scores)
        results[task_name] = {
            "score": avg_score,
            "seed_scores": seed_scores,
            "task_type": task_info["task_type"],
            "category": task_info["category"],
            "is_core": task_info["is_core"],
            "is_core_ext": task_info["is_core_ext"],
        }
    return results
