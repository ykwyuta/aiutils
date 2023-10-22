import utils.sk_helper as sh
import utils.chat_tools as ct
import asyncio

skHelper = sh.SkHelper()
urls = [
"https://devblogs.microsoft.com/dotnet/dotnet-framework-october-2023-security-and-quality-rollup-updates/",
]

import re

for url in urls:
    print(url)
    print("======================================")
    try:
        chunks = ct.get_url_content(url)
        keywords = skHelper.question_and_answer("\n".join(chunks))
        print(keywords)
        # result = skHelper.summarize_chunks(chunks)
        # print(result)
        # print("===================")
        # print(skHelper.translate(result))
        # print("===================")
        # print(skHelper.translate(skHelper.summarize_chunk("\n".join(chunks))))
    except Exception as e:
        print(e)
        print("Failed to summarize.")