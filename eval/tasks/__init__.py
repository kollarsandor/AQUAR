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
from .task_registry import (
    TASK_REGISTRY,
    CORE_TASKS,
    CORE_EXT_TASKS,
    get_all_tasks,
    get_core_tasks,
    get_core_ext_tasks,
    run_evaluation,
)
