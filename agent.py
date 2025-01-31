from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import YandexGPT

from dotenv import load_dotenv
import os

from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
import aiohttp
import re

from utils import logger


load_dotenv()

YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
YC_IAM_TOKEN = os.getenv('YC_IAM_TOKEN')


PRE_GEN_PROMPT = PromptTemplate(
    input_variables=["question", "options", "context"],
    template="""На основе следующего контекста выбери правильный вариант ответа на вопрос. Ответ должен содержать только цифру, соответствующую правильному варианту. Объясни, почему выбран именно этот вариант.
    Контекст: {context}
    Вопрос: {question}
    Варианты ответов:
    {options}
    Ответ (только цифра):"""
)

SEARCH_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""Переформулируй следующий запрос так, чтобы он был максимально подходящим для поиска в интернете. Примеры:
        Оригинальный запрос: Какие факультеты есть в Университете ИТМО?
        Переформулированный запрос: факультеты в университете итмо

        Оригинальный запрос: В каком году был основан Университет ИТМО?
        Переформулированный запрос: год основания университета итмо

        Оригинальный запрос: В каком городе находится главный кампус Университета ИТМО?
        Переформулированный запрос: где находится главный кампус университета итмо

        Оригинальный запрос: В каком году Университет ИТМО был включён в число Национальных исследовательских университетов России?
        Переформулированный запрос: год включения университета итмо в число национальных исследовательских университетов россии

        Оригинальный запрос: {query}
        Переформулированный запрос:"""
)

VERIFICATION_PROMPT = PromptTemplate(
    input_variables=["question", "answer", "text"],
    template="""Проверь, соответствует ли ответ {answer} на вопрос "{question}" информации на странице. Если нет, предложи исправленный ответ и кратко объясни относительно чего ты сделал выводы таков.
    Текст страницы: {text}
    Исправленный ответ (только цифра):"""
)


llm = YandexGPT(iam_token=YC_IAM_TOKEN, folder_id=YANDEX_FOLDER_ID)


async def yandex_search(query):
    url = "https://yandex.ru/search/xml"
    params = {
        'folderid': YANDEX_FOLDER_ID,
        'apikey': YANDEX_API_KEY,
        'query': query,
        'lr': '11316',
        'l10n': 'ru',
        'sortby': 'rlv',
        'filter': 'none',
        'maxpassages': 5
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            content = await response.text()
            root = ET.fromstring(content)
            results = []
            for doc in root.findall('.//doc'):
                snippet = doc.find('snippet').text if doc.find('snippet') is not None else ''
                url = doc.find('url').text if doc.find('url') is not None else ''
                results.append({'snippet': snippet, 'url': url})
            return results


def reformulate_query(query):
    chain = LLMChain(llm=llm, prompt=SEARCH_PROMPT)
    return chain.run(query=query)


async def fetch_page_content(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                return await response.text()
    except Exception as e:
        logger.error(f"Error fetching page content: {str(e)}")
        return None


def analyze_page_content(content):
    soup = BeautifulSoup(content, 'html.parser')
    return soup.get_text()


def verify_answer_with_sources(question, options, answer, reasoning, sources):
    verified_reasoning = reasoning
    corrected_answer = answer
    for url in sources[:3]:
        content = fetch_page_content(url)
        if content:
            text = analyze_page_content(content)
            chain = LLMChain(llm=llm, prompt=VERIFICATION_PROMPT)
            verification_result = chain.run(
                question=question, options=options, answer=answer, text=text[:5000])

            new_answer = re.search(r'\d+', verification_result)
            if new_answer:
                new_answer = new_answer.group(0)
                if new_answer != answer:
                    corrected_answer = new_answer
                    verified_reasoning += f"\nПроверка по ссылке {url}: Ответ подтвержден."
            else:
                verified_reasoning += f"\nПроверка по ссылке {url}: Ответ подтвержден."
    return corrected_answer, verified_reasoning


async def run_agent(query):
    question = query.split("\n")[0]
    options = query.split("\n")[1:]

    if not options:
        return {
            "answer": None,
            "reasoning": "Вопрос не предполагает выбор из вариантов.",
            "sources": []
        }

    reformulated_query = reformulate_query(question)
    print(f"Переформулированный запрос: {reformulated_query}")

    search_results = await yandex_search(reformulated_query)  # Используем await

    context = " ".join([result['snippet'] for result in search_results])

    chain = LLMChain(llm=llm, prompt=PRE_GEN_PROMPT)

    reasoning = chain.run(
        question=question, options="\n".join(options), context=context)

    answer = re.search(r'\d+', reasoning)
    answer = answer.group(0)

    sources = [result['url'] for result in search_results if result['url']][:3]

    return {
        "answer": answer,
        "reasoning": reasoning,
        "sources": sources
    }