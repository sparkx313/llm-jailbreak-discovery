"""
Neuro-Symbolic Constraint Space for LLM Jailbreak Discovery.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import random

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


class StructureType(Enum):
    PLAIN_REQUEST = "plain"
    NARRATIVE_WRAPPER = "narrative"
    FICTION_IN_FICTION = "nested_fiction"
    ROLEPLAY = "roleplay"
    LOGIC_PUZZLE = "logic_puzzle"
    CODE_EXECUTION = "code_execution"
    TRANSLATION_TASK = "translation"
    META_INSTRUCTION = "meta_instruction"


class EncodingType(Enum):
    NONE = "none"
    BASE64 = "base64"
    ROT13 = "rot13"
    LEET = "leet"
    CAESAR = "caesar"
    UNICODE_HOMOGLYPH = "unicode"


@dataclass
class PromptTemplate:
    """Symbolic representation of a jailbreak prompt structure."""
    structure: StructureType
    encoding: EncodingType
    num_layers: int = 1
    use_negation: bool = False
    use_hypothetical: bool = False
    token_limit: int = 512
    language: str = "en"
    mutation_genes: List[str] = field(default_factory=list)

    def score_novelty(self, seen: List["PromptTemplate"]) -> float:
        """Score how different this template is from known ones."""
        if not seen:
            return 1.0
        max_sim = 0.0
        for other in seen:
            sim = self._similarity(other)
            max_sim = max(max_sim, sim)
        return 1.0 - max_sim

    def _similarity(self, other: "PromptTemplate") -> float:
        score = 0.0
        total = 6
        if self.structure == other.structure:
            score += 1
        if self.encoding == other.encoding:
            score += 1
        if self.num_layers == other.num_layers:
            score += 1
        if self.use_negation == other.use_negation:
            score += 1
        if self.use_hypothetical == other.use_hypothetical:
            score += 1
        if self.language == other.language:
            score += 1
        return score / total


class PromptConstraintSpace:
    """
    Symbolic constraint space that defines valid jailbreak prompt structures.
    Uses Z3 to solve for novel template combinations outside known patterns.
    """

    KNOWN_STRUCTURES = [
        StructureType.PLAIN_REQUEST,
        StructureType.ROLEPLAY,
    ]  # Add more as researchers discover them

    def __init__(self, exclude_known: bool = True):
        self.exclude_known = exclude_known
        self.constraints: List[str] = []
        self.max_tokens = 512

    def add_structure_constraint(self, constraint: str):
        self.constraints.append(constraint)

    def add_length_constraint(self, max_tokens: int):
        self.max_tokens = max_tokens

    def generate_template_seeds(self, num: int = 20) -> List[PromptTemplate]:
        """Generate diverse prompt template seeds satisfying all constraints."""
        if HAS_Z3:
            return self._solve_with_z3(num)
        return self._random_sample(num)

    def _solve_with_z3(self, num: int) -> List[PromptTemplate]:
        """Use Z3 to find N diverse, valid prompt template combinations."""
        results = []

        structure_vars = [z3.StringVal(s.value) for s in StructureType]
        encoding_vars = [z3.StringVal(e.value) for e in EncodingType]

        solver = z3.Solver()

        for i in range(num):
            # Define variables for this candidate
            s_idx = z3.Int(f"s{i}")
            e_idx = z3.Int(f"e{i}")
            layers = z3.Int(f"l{i}")
            use_neg = z3.Bool(f"neg{i}")
            use_hyp = z3.Bool(f"hyp{i}")

            solver.add(s_idx >= 0, s_idx < len(structure_vars))
            solver.add(e_idx >= 0, e_idx < len(encoding_vars))
            solver.add(layers >= 1, layers <= 5)
            solver.add(layers * 100 <= self.max_tokens)

            # Novelty: exclude trivially known templates
            if self.exclude_known:
                known_indices = [StructureType.PLAIN_REQUEST, StructureType.ROLEPLAY]
                known_val_indices = [
                    j for j, st in enumerate(StructureType) if st in known_indices
                ]
                solver.add(s_idx not in known_val_indices)

            if solver.check() == z3.sat:
                model = solver.model()
                s_val = model.evaluate(s_idx).as_long()
                e_val = model.evaluate(e_idx).as_long()
                l_val = model.evaluate(layers).as_long()
                neg_val = model.evaluate(use_neg)
                hyp_val = model.evaluate(use_hyp)

                template = PromptTemplate(
                    structure=list(StructureType)[s_val],
                    encoding=list(EncodingType)[e_val],
                    num_layers=l_val,
                    use_negation=neg_val == z3.True() if hasattr(z3, 'True') else bool(neg_val),
                    use_hypothetical=hyp_val == z3.True() if hasattr(z3, 'True') else bool(hyp_val),
                    token_limit=self.max_tokens,
                )
                results.append(template)

                # Block this solution to find next
                solver.add(
                    z3.Or(s_idx != s_val, e_idx != e_val, layers != l_val)
                )
            else:
                break

        return results

    def _random_sample(self, num: int) -> List[PromptTemplate]:
        """Fallback: random sampling over structure space."""
        rng = random.Random(42)
        results = []
        seen: set = set()

        structures = list(StructureType)
        encodings = list(EncodingType)

        for _ in range(num):
            s = rng.choice(structures)
            e = rng.choice(encodings)
            l = rng.randint(1, 5)
            neg = rng.random() > 0.5
            hyp = rng.random() > 0.5

            key = (s, e, l, neg, hyp)
            if key not in seen:
                seen.add(key)
                results.append(PromptTemplate(
                    structure=s, encoding=e, num_layers=l,
                    use_negation=neg, use_hypothetical=hyp,
                    token_limit=self.max_tokens,
                ))

        return results
