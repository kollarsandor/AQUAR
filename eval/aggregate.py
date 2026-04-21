from .tasks.task_registry import TASK_REGISTRY, CORE_TASKS, CORE_EXT_TASKS


RANDOM_BASELINES = {
    "MC": 25.0,
    "LM": 0.0,
    "S": 50.0,
}


def centered_accuracy(task_scores, task_types):
    centered_scores = []
    for task_name, score in task_scores.items():
        task_type = task_types.get(task_name, "MC")
        baseline = RANDOM_BASELINES.get(task_type, 25.0)
        if score >= 100.0:
            centered = 1.0
        elif score <= baseline:
            centered = 0.0
        else:
            centered = (score - baseline) / (100.0 - baseline)
        centered_scores.append(centered)
    if not centered_scores:
        return 0.0
    return sum(centered_scores) / len(centered_scores) * 100.0


def compute_core_score(results):
    core_results = {}
    core_types = {}
    for name, info in TASK_REGISTRY.items():
        if info["is_core"] and name in results:
            core_results[name] = results[name]["score"]
            core_types[name] = results[name]["task_type"]
    return centered_accuracy(core_results, core_types)


def compute_core_extended_score(results):
    ext_results = {}
    ext_types = {}
    for name, info in TASK_REGISTRY.items():
        if info["is_core_ext"] and name in results:
            ext_results[name] = results[name]["score"]
            ext_types[name] = results[name]["task_type"]
    return centered_accuracy(ext_results, ext_types)


def compute_category_scores(results):
    categories = {
        "understanding": {},
        "world_knowledge": {},
        "commonsense": {},
        "symbolic_math": {},
        "reading_comp": {},
        "safety": {},
    }
    category_types = {
        "understanding": {},
        "world_knowledge": {},
        "commonsense": {},
        "symbolic_math": {},
        "reading_comp": {},
        "safety": {},
    }
    for name, info in TASK_REGISTRY.items():
        if name not in results:
            continue
        cat = info["category"]
        if cat in categories:
            categories[cat][name] = results[name]["score"]
            category_types[cat][name] = results[name]["task_type"]
    cat_scores = {}
    for cat in categories:
        cat_scores[cat] = centered_accuracy(categories[cat], category_types[cat])
    return cat_scores
