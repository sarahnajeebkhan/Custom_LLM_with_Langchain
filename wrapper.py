from langchain_core.language_models.llms import LLM
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Custom LLM Class
class LangLLM(LLM):
    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "LLM Wrapper for Langchain"

    def _call(self,prompt: str,stop: Optional[List[str]] = None,chatbot=None) -> str:

        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", device_map = 'cuda')
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", device_map = 'cuda')
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        generation_args = {
            "max_new_tokens": 100,
            "return_full_text": False,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 50
        }
        output = pipe(prompt, **generation_args)
        response = output[0]['generated_text']
        return response

# Instantiate the LangChain LLM
llm = LangLLM()

#Perform all/any langchain operations as per your requirements
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Answer the question: {question}"
)
memory = ConversationBufferMemory(llm=llm, max_token_limit=100)
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

# Query the model
conversation.predict(input="Hi. Have you read the poem 'dulce et decorum est'. What's your take on it?")
