import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BERTClassifier:
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        id2label: Optional[Dict[int, str]] = None,
    ):
        self.model_name = model_name

        if device is not None:
            self.device = device
        else:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        self.id2label = id2label or {0: "neutral", 1: "positive", 2: "negative"}
        self._load_model()

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        is_local_dir = Path(self.model_name).exists() and Path(self.model_name).is_dir()
        if is_local_dir:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=len(self.id2label)
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, top_k: int = 3) -> Dict:
        start = time.time()

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        k = max(1, min(int(top_k), probs.shape[0]))
        top_probs, top_idx = torch.topk(probs, k=k)

        preds = []
        for score, idx in zip(top_probs.tolist(), top_idx.tolist()):
            preds.append({"label": self.id2label.get(idx, str(idx)), "score": float(score)})

        return {"predictions": preds, "processing_time": time.time() - start}

    def predict_batch(self, texts: List[str], top_k: int = 3) -> List[Dict]:
        return [self.predict(t, top_k=top_k) for t in texts]

