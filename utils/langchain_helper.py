from langchain.tools import DuckDuckGoSearchRun
from typing import Optional
from langchain.callbacks.manager import (CallbackManagerForToolRun)

class DuckDuckGoCustomDomainSearchRun(DuckDuckGoSearchRun):

    def __init__(self, domain: str) -> None:
        self.domain = domain
        super().__init__()

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if len(query) != 0:
            query = "{0} site:{1}".format(query, self.domain)
        return super()._run(query)
