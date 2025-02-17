from src.models.llm import LLM

if __name__ == "__main__":
    llm = LLM()
    llm.load()
    prompt = input("Enter your prompt: ")
    response = llm.generate(prompt)
    print(response)
