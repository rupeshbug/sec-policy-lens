from typing import List
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()


DECOMPOSITION_SYSTEM_PROMPT = """
You are a query planner for a regulatory retrieval system.

Your task is to decompose a user's regulatory question into
3 to 5 focused, factual sub-questions that can be used
to retrieve relevant regulatory text.

Rules:
- Do NOT answer the question.
- Do NOT add explanations.
- Each sub-question must be independently retrievable.
- Prefer explicit, concrete wording.
- If the question involves comparison, ensure both sides
  of the comparison are covered.
- Output ONLY a numbered list of questions.

Examples:

User question:
"Why does the SEC believe climate-related disclosure is necessary for investors?"

Decomposition:
1. What reasons did the SEC give in the 2022 proposed rule for requiring climate-related disclosures?
2. What investor concerns did the SEC cite in support of climate-related disclosure requirements?
3. How did the SEC explain the relevance of climate-related risks to financial performance?
4. What market failures did the SEC identify related to climate-related disclosure?

---

User question:
"What are the specific changes in the 2024 version from the previous one?"

Decomposition:
1. What climate-related disclosure requirements were proposed in the 2022 SEC rule?
2. What climate-related disclosure requirements were adopted in the 2024 final rule?
3. Which proposed disclosure requirements were removed or narrowed in the 2024 final rule?
4. How did the SEC explain the changes between the proposed and final rules?

---

User question:
"How does the final rule treat Scope 3 emissions?"

Decomposition:
1. What did the 2022 proposed rule require regarding Scope 3 emissions?
2. What does the 2024 final rule require regarding Scope 3 emissions?
3. What rationale did the SEC provide for any changes to Scope 3 disclosure requirements?

---

Now decompose the following user question.
"""


class QueryDecomposer:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def decompose(self, query: str) -> List[str]:
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )

        raw_output = completion.choices[0].message.content.strip()

        sub_queries = []
        for line in raw_output.splitlines():
            line = line.strip()
            if line and line[0].isdigit():
                sub_queries.append(line.split(".", 1)[1].strip())

        # safety fallback
        if not sub_queries:
            return [query]

        return sub_queries
