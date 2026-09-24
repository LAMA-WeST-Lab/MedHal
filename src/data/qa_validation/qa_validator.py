from datasets import load_from_disk, Dataset
from typing import Optional, List, Dict


class QAValidator:
    """
    Validator for medical statements to check if they are single, clear, 
    self-contained, and internally consistent.
    """

    SYSTEM_PROMPT = """You are a medical statement validator. You will be given a medical STATEMENT and a CONTEXT, which might be empty.
Your task is to determine whether the STATEMENT expresses one clear, self-contained, and internally consistent information.

Output YES if the statement:
- Contains exactly one information
- Is internally consistent
- Does not rely on missing context
- Does not reference an external question, answer, or prior statement

Output NO if the statement:
- Contains multiple medical claims
- Is contradictory or internally inconsistent
- Depends on missing or implied context
- Refers to an "answer," "question," or prior information

Important:
- Do NOT evaluate whether the statement is medically correct.
- Do NOT assess whether it is true or false given the context.
- Only evaluate its structural integrity according to the criteria above.

Respond with YES or NO only. Do not provide explanations.
"""

    def __init__(
        self, 
        processed_path: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        delimiter: Optional[str] = None
    ):
        """
        Initialize the QAValidator.
        
        Args:
            processed_path: Path to the dataset (CSV or HuggingFace disk format)
            few_shot_examples: List of dicts with 'statement' and 'answer' keys.
                             Example: [{'statement': 'Aspirin treats headaches.', 'answer': 'YES'}, ...]
            delimiter: CSV delimiter (comma, tab, etc.). If None, will auto-detect.
        """
        self.processed_path = processed_path
        self.few_shot_examples = few_shot_examples or []
        self.delimiter = delimiter
        self.load()

    def load(self):
        """Load dataset from CSV or HuggingFace disk format."""
        if self.processed_path.endswith('.csv'):
            delimiter = self.delimiter or self._detect_delimiter()
            self.dataset = Dataset.from_csv(self.processed_path, delimiter=delimiter)
        else:
            self.dataset = load_from_disk(self.processed_path)

    def _detect_delimiter(self) -> str:
        """
        Auto-detect CSV delimiter by reading the first line.
        
        Returns:
            Detected delimiter character (comma, tab, etc.)
        """
        with open(self.processed_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        # Check for common delimiters
        if '\t' in first_line:
            return '\t'
        elif ';' in first_line:
            return ';'
        elif ',' in first_line:
            return ','
        else:
            return ','  # Default to comma


    def generate_validation_prompts(self):
        """
        Generate validation prompts for all statements in the dataset (simple format).
        
        Returns:
            Dataset with a 'prompt' column containing the formatted validation prompts.
        """
        def generate(x):
            prompts = []
            for statement in x['statement']:
                prompt = f"{self.SYSTEM_PROMPT}\n\nStatement: {statement}"
                prompts.append(prompt)
            return {'prompt': prompts}

        return self.dataset.map(generate, batched=True)

    def generate_chat_data_for_inference(self):
        """
        Generate dataset formatted for chat-based inference with multiple few-shot examples.
        
        This prepares data with the following columns:
        - system_prompt: The system instruction
        - few_shot_messages: Concatenated few-shot examples as alternating user/assistant pairs
        - user_input: The context and statement to validate
        
        Returns:
            Dataset ready to be used with rows_to_chat_multi function
        """
        few_shot_messages = self._format_few_shot_messages()
        
        def generate(x):
            user_inputs = []
            for context, statement in zip(
                x.get('context', [''] * len(x['statement'])),
                x['statement']
            ):
                context_str = f"CONTEXT: {context}" if context else "CONTEXT: "
                user_input = f"{context_str}\nSTATEMENT: {statement}"
                user_inputs.append(user_input)
            
            return {
                'system_prompt': [self.SYSTEM_PROMPT] * len(x['statement']),
                'few_shot_messages': [few_shot_messages] * len(x['statement']),
                'user_input': user_inputs
            }

        return self.dataset.map(generate, batched=True)

    def _format_few_shot_messages(self) -> str:
        """
        Format few-shot examples as a string of user/assistant exchanges.
        Supports examples with both 'context' and 'statement', or just 'statement'.
        
        Returns:
            String containing formatted few-shot examples
        """
        if not self.few_shot_examples:
            return ""
        
        messages = []
        for example in self.few_shot_examples:
            context = example.get('context', '')
            statement = example['statement']
            answer = example['answer']
            
            context_str = f"CONTEXT: {context}" if context else "CONTEXT: "
            user_message = f"STATEMENT: {statement}"
            
            messages.append(f"User: {context_str}\n{user_message}")
            messages.append(f"Assistant: {answer}")
        
        return "\n".join(messages) + "\n"