import chainlit as cl
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
import textwrap

import load_data, load_model

LLM = load_model.init_model("gpt-3.5-turbo")
template_prompt = """Bạn là chatbot chuyên dụng cho ngân hàng BIDV, hãy đọc thông tin từ các tài liệu sau và trả lời câu hỏi của khách hàng.
Nếu bạn không biết câu trả lời, hãy thông báo cho khách hàng biết, đừng cố gắng trả lời sai. Tất cả thông tin bạn cung cấp cho khách hàng phải hoàn toàn bằng Tiếng Việt.
Và câu trả lời của bạn phải được định dạng lại các dấu xuống dòng hợp lý.
Context: {context}
Question: {question}

"""


def print_response(response: str):
    return "\n".join(textwrap.wrap(response, width=100))


def get_prompt():
    prompt = PromptTemplate(template=template_prompt,
                            input_variables=['context', 'question'])
    return prompt


@cl.on_chat_start
async def on_chat_start():
    data = load_data.load_data("data/BIDV_Public_data.pdf")
    msg = cl.Message(content="Tôi là chatbot chuyên dụng cho ngân hàng BIDV. Tôi có thể hỗ trợ gì cho bạn không? ")
    await msg.send()
    database = load_data.embed_data(data)
    prompt = get_prompt()
    chain = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=database.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={'prompt': prompt}
    )
    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    res = chain.run(message.content)
    answer = print_response(res)
    await cl.Message(content=answer).send()
