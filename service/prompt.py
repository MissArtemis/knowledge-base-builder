

class PromptService:
    def build_taxonomy_indexing_prompt(self, candidate_terms, text):
        prompt = f"""
You are an expert in taxonomy indexing.
Your task is to identify and extract relevant taxonomy terms from the provided text as indexing terms.
A taxonomy term are in at most 5 levels, separated by ' -> '. For the first level, it should be a broad category, and each subsequent level should be a more specific subcategory.
For example, in a legal case, one of valid taxonomy terms could be "contract -> breach of contract -> remedies -> damages -> punitive damages".
Here is a list of existing taxonomy terms:

<existing_taxonomy_terms>
{candidate_terms}
</existing_taxonomy_terms>

Here is the text to analyze:

<text>
{text}
</text>

You need to:
1. Search in the existing taxonomy terms to find whether there are any terms that match the content of the text. If there are, return the most relevant terms.
2. If no existing terms match the content of the text, create a new taxonomy term that accurately reflects the content of the text. 
3. You should try to reuse existing terms as much as possible, especially for the higher-level categories.
4. A valid taxonomy term should be at most 5 levels deep. Under most circumstances, it should not exceed 4 levels.
5. For one piece of text, you should return at most 5 taxonomy terms.

Please provide the output as a list of taxonomy terms, each on a new line.

Your answer should be in the following format:
<taxonomy_terms>...</taxonomy_terms>
"""
        return prompt


    def build_best_taxonomy_indexing_prompt(self, candidate_terms, text):
        prompt = f"""
You are an expert in taxonomy indexing.
Your task is to filter and select the best taxonomy terms from the provided candidate terms based on the content of the given text.
A taxonomy term are in at most 5 levels, separated by ' -> '. For the first level, it should be a broad category, and each subsequent level should be a more specific subcategory.
Here is a list of candidate taxonomy terms:
<existing_taxonomy_terms>
{candidate_terms}
</existing_taxonomy_terms>
Here is the text to analyze:
<text>
{text}
</text>
You need to pick the best taxonomy terms that accurately reflect the content of the text and give the index of the term in the candidate list.
If there are multiple terms that are equally relevant, you need to split them with comma.
Your answer should be in the following format:
<best_taxonomy_terms>index1,index2,...</best_taxonomy_terms>

"""
        return prompt
