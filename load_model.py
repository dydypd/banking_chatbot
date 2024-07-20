from langchain.llms import OpenAI


def init_model(model_name):
    OPENAI_API_KEY = "YOUR_API_KEY"
    model = OpenAI(temperature=0, model_name=model_name, api_key=OPENAI_API_KEY)
    return model
