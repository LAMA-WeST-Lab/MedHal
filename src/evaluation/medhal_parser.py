from __future__ import annotations
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import re


class MedHalParser:
    """
    Evaluator that computes precision, recall, f1 scores on medhal dataset based on given generations.
    It also computes the embedding similarity (cosine) of the explanations.
    """

    VALID_PATTERN = r'Factual(?::)?(?:\n| |\*)*(YES|NO)'
    EXPLANATION_PATTERN = r'### Explanation([\s\S]+)'
    YES_PATTERN = r'(?:Factual)?(?:\s|\:)(?:\[)?(?:yes|is factual)(?:\])?(?:\s|\:)?'
    NO_PATTERN = r'(?:Factual)?(?:\s|\:)(?:\[)?(?:no|not factual)(?:\])?(?:\s|\:)?'

    DATA_TO_TASK = {
        'acm': 'Information Extraction',
        'medmcqa': 'Question-Answering',
        'medqa': 'Question-Answering',
        'mednli': 'NLI',
        'sumpubmed': 'Summarization'
    }

    def __init__(self) -> None:
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def load(self, dataset: str | Dataset):
        if isinstance(dataset, str):
            return Dataset.from_csv(dataset)
        else:
            return dataset

    def parse(self, dataset: str | Dataset, add_prompt: bool, output_col: str = 'OUTPUT', prompt_col: str = 'text'):
        data = self.load(dataset)
        data = data.map(
            self._get_prediction,
            fn_kwargs={
                'add_prompt': add_prompt,
                'output_col': output_col,
                'prompt_col': prompt_col,
            },
            desc='Parsing answers'
        )
        return data

    def evaluate(self, dataset: str | Dataset, add_prompt: bool, output_col: str = 'OUTPUT', prompt_col: str = 'text'):
        data = self.load(dataset)
        data = data.map(
            self._get_prediction,
            fn_kwargs={
                'add_prompt': add_prompt,
                'output_col': output_col,
                'prompt_col': prompt_col,
            },
            desc='Parsing answers',
        )

        print('Total data size : ', len(data))

        filtered = data.filter(lambda x: x['valid'], desc='Filtering invalid samples')
        print('Filtered size : ', len(filtered))
        invalid = data.filter(lambda x: not x['valid'], desc='Retrieving invalid samples')
        print('Invalid size : ', len(invalid))

        y_pred = filtered['prediction']
        y_test = filtered['label']

        # We only evaluate explanation similarity if the model predicted False
        # and the true factual label is False. Otherwise, the explanation does
        # not make sense.
        valid_explanation = filtered.filter(lambda x: x['prediction'] == x['label'] and not x['label'], desc='Retrieving valid explanations')

        embsim = self._get_embsim_scores(valid_explanation['explanation'], valid_explanation['explanation_gen'])

        valid_explanation = valid_explanation.add_column('embsim', embsim)

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'embsim': np.mean(embsim),
            'invalid': invalid,
            'valid': filtered,
            'valid_explanation': valid_explanation,
        }

    def show_stats(self, results):
        return results['valid'].to_pandas().groupby('source').size()

    def _get_prediction(self, row, add_prompt: bool, output_col: str, prompt_col: str):
        if row[output_col] is None:
            return {
                'valid': False,
                'prediction': False,
                'explanation_gen': ''
            }

        if add_prompt:
            output = row[prompt_col] + row[output_col]
        else:
            output = row[output_col]

        if not output:
            return {
                'valid': False,
                'prediction': False,
                'explanation_gen': ''
            }

        output = output.lower()
        match = re.search(r'\b(yes|no)\b', output, re.IGNORECASE)

        if match:
            explanation_gen = output[match.end():].strip()
            prediction = match.group(0).lower() == 'yes'
            return {
                'valid': True,
                'prediction': prediction,
                'explanation_gen': explanation_gen
            }

        return {
            'valid': False,
            'prediction': False,
            'explanation_gen': ''
        }

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """Embeding Similarity"""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def _get_embsim_scores(self, references, predictions):
        refs = ["" if x is None else str(x) for x in references]
        preds = ["" if x is None else str(x) for x in predictions]

        assert len(refs) == len(preds)

        E_ref = self.embedder.encode(refs, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        E_pred = self.embedder.encode(preds, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

        scores = (E_ref * E_pred).sum(axis=1)  # cosine, since normalized
        return scores.tolist()

    @staticmethod
    def plot_model_comparison_accuracy(paths: list, model_names: list, add_prompts: list, category_column: str = 'source'):
        """
        Plots the accuracy score of multiple models across all categories with lines connecting dots.

        Args:
            paths (list): List of paths to the dataset results for each model.
            model_names (list): List of model names corresponding to the paths.
            add_prompts: List of boolean variables indicating whether for each path, the prompt must be added
            category_column (str): The column in the dataset that represents the category.
        """
        if len(paths) != len(model_names) or len(add_prompts) != len(paths):
            raise ValueError("The number of paths, model names and add_prompts variables must be the same.")

        accuracies = {}
        all_categories = set()

        for path, model_name, add_prompt in zip(paths, model_names, add_prompts):
            evaluator = MedHalParser()
            results = evaluator.evaluate(dataset=path, add_prompt=add_prompt, output_col='output')
            print(results)
            df = results['valid'].to_pandas()

            def map_categories(category):
                return MedHalParser.DATA_TO_TASK[category]

            df[category_column] = df[category_column].apply(map_categories)

            category_accuracies = df.groupby(category_column).apply(
                lambda x: f1_score(x['label'], x['prediction'])  # sum(x['prediction'] == x['label']) / len(x) * 100
            ).to_dict()

            accuracies[model_name] = category_accuracies
            all_categories.update(category_accuracies.keys())

        all_categories = sorted(list(all_categories))

        # Prepare data for plotting
        model_accuracies = {model_name: [accuracies[model_name].get(category, 0) for category in all_categories] for model_name in model_names}

        # Create the plot
        plt.figure(figsize=(12, 6))

        for model_name in model_names:
            plt.plot(all_categories, model_accuracies[model_name], marker='o', label=model_name)

        plt.xlabel('Task', fontsize=12)
        plt.ylabel('F1-Score', fontsize=12)
        plt.legend()
        plt.show()
