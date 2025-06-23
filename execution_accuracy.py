# /custom_metrics/execution_accuracy.py
import os
import sqlite3
import asyncio
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class ExecutionAccuracyMetric(BaseMetric):
    def __init__(self, db_dir: str, threshold: float = 1.0):
        # Definimos o _threshold que a propriedade 'threshold' da classe mãe espera encontrar.
        self._threshold = threshold
        self.db_dir = db_dir

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            db_path = os.path.join(self.db_dir, test_case.input, f"{test_case.input}.sqlite")
            if not os.path.exists(db_path):
                return 0.0
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(test_case.actual_output)
            predicted_results = set(cursor.fetchall())
            
            cursor.execute(test_case.expected_output)
            expected_results = set(cursor.fetchall())
            
            conn.close()
            
            score = 1.0 if predicted_results == expected_results else 0.0
            return score
        except Exception:
            # Se qualquer erro de SQL ocorrer, o score é 0.
            if 'conn' in locals() and conn:
                conn.close()
            return 0.0

    async def a_measure(self, test_case: LLMTestCase) -> float:
        # A versão assíncrona é necessária e pode simplesmente chamar a síncrona.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.measure, test_case)

    def is_successful(self) -> bool:
        # Este método é chamado pelo framework DEPOIS de ele executar 'measure'
        # e atualizar o atributo `self.score` com o resultado.
        if self.score is None:
            # Isso pode acontecer se a avaliação falhar por algum motivo.
            return False
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Execution Accuracy"