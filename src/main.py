import os
import httpx
import json
import fitz
# kor
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text

# Langchain
from langchain_openai import ChatOpenAI

# Custom libraries
from src import credentials, examples

DOSSIER_NAME = '99-ru-2005'

def read_txt_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# Write everything to a text file except first and last page
def dossier_pdf_to_txt():
    with fitz.open(os.path.join(credentials.data_path, 'pdf-files', f'{DOSSIER_NAME}.pdf')) as doc:
        text = ""
        for page in doc:
            if page.number == 0 or page.number == doc.page_count - 1:
                continue
            text += page.get_text()
    with open(os.path.join(credentials.data_path, 'text-files', f'{DOSSIER_NAME}.txt'), 'w', encoding='utf-8') as file:
        file.write(text)


def main():
    dossier_pdf_to_txt()
    pdf_as_txt = read_txt_from_file(os.path.join(credentials.data_path, 'text-files', f'{DOSSIER_NAME}.txt'))
    text_flat = pdf_as_txt.replace('\n', ' ')
    with open(os.path.join(credentials.data_path, 'text-files', f'{DOSSIER_NAME}_flat.txt'), 'w',
              encoding='utf-8') as file:
        file.write(text_flat)

    # Set up LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-1106",
        temperature=0,
        max_tokens=4096,
        openai_api_key=credentials.openai_api_key,
        http_client=httpx.Client(proxies=credentials.proxies)
    )

    # Set up Kor
    schema = Object(
        id="dossier",
        description=(
            "A public dossier issued by the Basel-Stadt government (written in German) "
            "contains relevant statistics presented through four to five articles. "
            "Each article in the dossier features a title followed by a summary "
            "followed by author initials followed by text."
        ),
        attributes=[
            Text(
                id="title",
                description="Title of an article in the dossier. Can be several expressions.",
                required=True,
                many=True,
                # examples=examples.title
            ),
            Text(
                id="summary",
                description=("Summary of the articles with the most important statistical numbers and conclusions."
                             "Can be several sentences."),
                required=True,
                many=True,
                # examples=examples.subtitle
            ),
            Text(
                id="author",
                description=("Author of an article in the dossier. The authors are credited at the end of the subtitle "
                             "using their initials, typically the first letter of the first name followed "
                             "by the first letter  of the last name."),
                required=True,
                many=True,
                # examples=examples.author
            )
        ]
    )

    chain = create_extraction_chain(llm, schema, encoder_or_encoder_class='json')
    # print(chain.prompt.format_prompt(text="[user input]").to_string()); quit()
    result = chain.run(text_flat)['data']

    print(result)

    # Save result as json
    with open(os.path.join(credentials.data_path, 'results', f'{DOSSIER_NAME}_extract.json'), 'w',
              encoding='utf-8') as file:
        json.dump(result, file, indent=4)


if __name__ == '__main__':
    main()

'''
Text(
    id="summary",
    description=("Summary of an article at the beginning. "
                 "Several sentences. Ends with 'mehr auf Seite X'."),
    required=True,
    many=True,
    #examples=examples.summary
),
'''