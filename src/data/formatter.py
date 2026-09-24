from typing import Any, Dict, List
from src.models.prometheus import Prometheus
from src.models.openbiollm import OpenBioLLM

TASK_DESCRIPTION = """### Task Description
- You will evaluate whether a medical statement is factually accurate.
- The statement may reference a provided context.
- Respond with "YES" if the statement is factually correct or "NO" if it contains inaccuracies.
- In order to answer YES, everything in the statement must be supported by the context.
- In order to answer NO, there must be at least one piece of information in the statement that is not supported by the context.
"""

CONTEXT = """### Context
{context}
"""

MEDHAL_FORMAT_TRAINING_CONTEXT = TASK_DESCRIPTION + "\n" + CONTEXT + """### Statement
{statement}

### Factual
{label}

### Explanation
{explanation}
"""

MEDHAL_FORMAT_TRAINING_NO_CONTEXT = TASK_DESCRIPTION + """
### Statement
{statement}

### Factual
{label}

### Explanation
{explanation}
"""

MEDHAL_FORMAT_INFERENCE_CONTEXT = TASK_DESCRIPTION + "\n" + CONTEXT + """### Statement
{statement}

### Factual
"""

MEDHAL_FORMAT_INFERENCE_NO_CONTEXT = TASK_DESCRIPTION + """
### Statement
{statement}

### Factual
"""

# Formats for few-shot examples (without task description to avoid repetition)
MEDHAL_FORMAT_FEWSHOT_EXAMPLE_CONTEXT = CONTEXT + """### Statement
{statement}

### Factual
{label}

### Explanation
{explanation}
"""

MEDHAL_FORMAT_FEWSHOT_EXAMPLE_NO_CONTEXT = """### Statement
{statement}

### Factual
{label}

### Explanation
{explanation}
"""

MEDHAL_FORMAT_FEWSHOT_TARGET_CONTEXT = CONTEXT + """### Statement
{statement}

### Factual
"""

MEDHAL_FORMAT_FEWSHOT_TARGET_NO_CONTEXT = """### Statement
{statement}

### Factual
"""

# ---------------------------------------------------------------------------
# HallOumi template helpers
# ---------------------------------------------------------------------------
# The format uses structured tags expected by the HallOumi model:
#
#   <|context|>
#     <|s1|><sentence 1><end||s>
#     <|s2|><sentence 2><end||s>
#   <end||context>
#
#   <|request|><task instructions><end||request>
#
#   <|response|>
#     <|r1|><response sentence 1><end||r>
#     <|r2|><response sentence 2><end||r>
#   <end||response>
#
# For inference the <|response|> block is omitted so the model completes it.
# ---------------------------------------------------------------------------

HALLOUMI_TASK_INSTRUCTIONS = (
    "Evaluate whether the following medical statement is factually accurate "
    "given the provided context. "
    "Respond with 'YES' if the statement is fully supported by the context, "
    "or 'NO' if it contains at least one inaccuracy or unsupported claim. "
    "Then provide a brief explanation of your decision."
)


def _build_halloumi_context_block(context: str) -> str:
    """
    Wrap a plain-text context into the <|context|> structured block.
    Each sentence (split on '. ') becomes its own <|sN|> tag.
    If context is empty/None an empty block is returned.
    """
    if not context or context in ('None', ''):
        return "<|context|><end||context>"

    # Split into sentences naively on '. ' boundaries; keep trailing punctuation.
    raw_sentences = [s.strip() for s in context.replace('\n', ' ').split('. ') if s.strip()]
    # Re-attach the period that was consumed during splitting (except the last one
    # if it already ends with punctuation).
    sentences = []
    for idx, s in enumerate(raw_sentences):
        is_last = idx == len(raw_sentences) - 1
        if not is_last and not s.endswith(('.', '!', '?')):
            s = s + '.'
        sentences.append(s)

    parts = ["<|context|>"]
    for i, sent in enumerate(sentences, start=1):
        parts.append(f"<|s{i}|><{sent}><end||s>")
    parts.append("<end||context>")
    return "".join(parts)


def _build_halloumi_request_block(statement: str) -> str:
    """Wrap the task instruction + statement into the <|request|> block."""
    content = f"{HALLOUMI_TASK_INSTRUCTIONS} Statement: {statement}"
    return f"<|request|><{content}><end||request>"


def _build_halloumi_response_block(label: bool, explanation: str) -> str:
    """
    Build a complete <|response|> block for training.

    r1 → YES/NO verdict
    r2 → explanation
    """
    yes_no = "YES" if label else "NO"
    expl = explanation or ""
    return (
        f"<|response|>"
        f"<|r1|><{yes_no}><end||r>"
        f"<|r2|><{expl}><end||r>"
        f"<end||response>"
    )


def _build_halloumi_response_prefix() -> str:
    """Partial response block used as a prompt prefix during inference."""
    return "<|response|><|r1|><"


class Formatter:

    def __init__(self, tokenizer, training=True, model_wrapper='medhal'):
        self.tokenizer = tokenizer
        self.training = training
        self.model_wrapper = model_wrapper

        valid_wrappers = [
            'medhal', 'halloumi',
            'prometheus', 'openbiollm',
        ]
        if model_wrapper not in valid_wrappers:
            raise ValueError(
                f"Unsupported model_wrapper: {model_wrapper}. "
                f"Must be one of: {', '.join(valid_wrappers)}"
            )

    def __call__(self, x) -> Dict[str, str] | Dict[str, List[str]]:
        if isinstance(x['statement'], str):
            return {'text': self.format_sample(x['context'], x['statement'], x.get('label'), x.get('explanation'))}

        return {'text': self.format_batched(x)}

    def format_batched(self, samples: Dict[str, List[str]]) -> List[str]:
        if samples['statement'] is None:
            return []

        output_texts = []
        for i in range(len(samples['statement'])):
            context = samples['context'][i]
            statement = samples['statement'][i]
            label = None
            explanation = None

            if self.training:
                label = samples['label'][i]
                explanation = samples['explanation'][i] if samples['explanation'][i] else ''

            output_texts.append(self.format_sample(context, statement, label, explanation))

        return output_texts

    def format_sample(self, context, statement, label=None, explanation=None) -> str:
        if self.model_wrapper == 'halloumi':
            return self._format_halloumi_sample(context, statement, label, explanation)

        elif self.model_wrapper == 'prometheus':
            instruction = (
                f"Evaluate the factuality of this statement about the given context.\n"
                f"Context : {context}\n"
                f"Statement : {statement}"
            )
            return Prometheus.create_prompt(
                instruction=instruction,
                response_a="The statement is factual",
                response_b="The statement is not factual",
                rubric_type='factuality'
            )

        elif self.model_wrapper == 'openbiollm':
            if self.training and label is not None:
                yes_no_label = self.get_formatted_label(label)
                text = (
                    f"{OpenBioLLM.SYSTEM_PROMPT}\n\n"
                    f"{TASK_DESCRIPTION}"
                    f"### Context\n{context or ''}\n\n"
                    f"### Statement\n{statement}\n\n"
                    f"### Factual\n{yes_no_label}\n\n{explanation or ''}"
                )
                return text + self.tokenizer.eos_token
            else:
                return (
                    f"{OpenBioLLM.SYSTEM_PROMPT}\n\n"
                    f"### Context\n{context or ''}\n\n"
                    f"### Statement\n{statement}\n\n"
                    f"### Factual"
                )

        else:  # medhal
            return self._format_medhal_sample(context, statement, label, explanation)

    # ------------------------------------------------------------------
    # HallOumi
    # ------------------------------------------------------------------

    def _format_halloumi_sample(
        self,
        context,
        statement,
        label=None,
        explanation=None,
    ) -> str:
        """
        Format a sample using the HallOumi structured-tag template.

        Training output (label provided):
            <|context|>…<end||context>
            <|request|><task + statement><end||request>
            <|response|><|r1|><YES/NO><end||r><|r2|><explanation><end||r><end||response>
            <eos>

        Inference output (no label):
            <|context|>…<end||context>
            <|request|><task + statement><end||request>
            <|response|><|r1|><          ← model completes from here
        """
        context_block = _build_halloumi_context_block(context)
        request_block = _build_halloumi_request_block(statement)

        if self.training and label is not None:
            response_block = _build_halloumi_response_block(label, explanation)
            return context_block + request_block + response_block + self.tokenizer.eos_token
        else:
            return context_block + request_block + _build_halloumi_response_prefix()

    # ------------------------------------------------------------------
    # MedHAL
    # ------------------------------------------------------------------

    def _format_medhal_sample(self, context, statement, label=None, explanation=None) -> str:
        yes_no_label = self.get_formatted_label(label)

        if self.training and not label:
            assert explanation is not None, (
                f'When generating training samples, an explanation must be provided.\n'
                f'Statement : {statement}\nLabel : {yes_no_label}'
            )

        if self.training:
            med_hal_format_context = MEDHAL_FORMAT_TRAINING_CONTEXT
            med_hal_format_no_context = MEDHAL_FORMAT_TRAINING_NO_CONTEXT
        else:
            med_hal_format_context = MEDHAL_FORMAT_INFERENCE_CONTEXT
            med_hal_format_no_context = MEDHAL_FORMAT_INFERENCE_NO_CONTEXT

        if context is not None and context != 'None' and context != '':
            output = med_hal_format_context.format(
                context=context, statement=statement,
                label=yes_no_label, explanation=explanation,
            )
        else:
            output = med_hal_format_no_context.format(
                statement=statement, label=yes_no_label, explanation=explanation,
            )

        if self.training:
            output += self.tokenizer.eos_token

        return output

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def get_formatted_label(self, label):
        return 'YES' if label else 'NO'

    # ------------------------------------------------------------------
    # Few-shot helpers (MedHAL format)
    # ------------------------------------------------------------------

    def format_fewshot_example(self, context, statement, label, explanation) -> str:
        """Format a few-shot example WITHOUT task description (for use in multi-turn conversations)"""
        yes_no_label = self.get_formatted_label(label)

        if context is not None and context != 'None' and context != '':
            output = MEDHAL_FORMAT_FEWSHOT_EXAMPLE_CONTEXT.format(
                context=context,
                statement=statement,
                label=yes_no_label,
                explanation=explanation or '',
            )
        else:
            output = MEDHAL_FORMAT_FEWSHOT_EXAMPLE_NO_CONTEXT.format(
                statement=statement,
                label=yes_no_label,
                explanation=explanation or '',
            )

        return output.rstrip()

    def format_fewshot_target(self, context, statement) -> str:
        """Format the target question for few-shot WITHOUT task description"""
        if context is not None and context != 'None' and context != '':
            output = MEDHAL_FORMAT_FEWSHOT_TARGET_CONTEXT.format(
                context=context,
                statement=statement,
            )
        else:
            output = MEDHAL_FORMAT_FEWSHOT_TARGET_NO_CONTEXT.format(
                statement=statement,
            )

        return output.rstrip()

    def format_one_shot_with_sample(
        self,
        context_one_shot,
        statement_one_shot,
        label_one_shot,
        explanation_one_shot,
        context,
        statement,
    ) -> str:
        formatted_sample = MEDHAL_FORMAT_INFERENCE_CONTEXT.format(context=context, statement=statement)
        one_shot = self.format_one_shot(context_one_shot, statement_one_shot, label_one_shot, explanation_one_shot) + '\n'
        return one_shot + formatted_sample

    def format_one_shot(
        self,
        context_one_shot,
        statement_one_shot,
        label_one_shot,
        explanation_one_shot,
    ) -> str:
        return MEDHAL_FORMAT_TRAINING_CONTEXT.format(
            context=context_one_shot,
            statement=statement_one_shot,
            label=self.get_formatted_label(label_one_shot),
            explanation=explanation_one_shot,
        )

    def format_one_shot_example_only(
        self,
        context_one_shot,
        statement_one_shot,
        label_one_shot,
        explanation_one_shot,
    ) -> str:
        """Format a few-shot example WITHOUT the task description (to avoid repetition)"""
        yes_no_label = self.get_formatted_label(label_one_shot)

        if context_one_shot is not None and context_one_shot != 'None' and context_one_shot != '':
            template = CONTEXT + """### Statement
{statement}

### Factual
{label}

### Explanation
{explanation}
"""
        else:
            template = """### Statement
{statement}

### Factual
{label}

### Explanation
{explanation}
"""
        return template.format(
            context=context_one_shot,
            statement=statement_one_shot,
            label=yes_no_label,
            explanation=explanation_one_shot,
        )