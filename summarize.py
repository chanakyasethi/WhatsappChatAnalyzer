from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI

open_api_key="OPENAI_API_KEY"
llm=ChatOpenAI(model_name='gpt-3.5-turbo',openai_api_key=open_api_key)


def get_summary(text):
    generic_template='''
    Write a summary of the following speech:
    Speech : `{speech}`
    Translate the precise summary to {language}.

    '''
    prompt=PromptTemplate(
        input_variables=['speech','language'],
        template=generic_template
    )

    # complete_prompt=prompt.format(speech=text,language='English')

    llm_chain=LLMChain(llm=llm,prompt=prompt)
    summary=llm_chain.run({'speech':text,'language':'english'})
    return summary