from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        self.token_cache = {}
        print(f"Loaded model ({self.model_name}) on {self.device}")

    def load(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device)

    def tokenize(self, text: str) -> dict:
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded first")

        if text in self.token_cache:
            return self.token_cache[text]

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True,
        )
        self.token_cache[text] = tokens
        return tokens

    def generate(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        self.model.eval()

        inputs = self.tokenize(prompt)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
