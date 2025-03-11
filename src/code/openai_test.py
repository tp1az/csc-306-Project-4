from openai import OpenAI
client = OpenAI(api_key= "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") 
MESSAGE_SYSTEM_CONTENT = ("You are an assistant tasked with answering the questions asked of a given CSV in JSON format. Your response should follow this guidlines {'Answer': answer}"
+ "answer: answer using information from the provided CSV only. The answer should be one of these categories: "
+ "Boolean, Valid answers include True/False, Y/N, Yes/No (all case insensitive), "
+ "Category, A value from a cell (or a substring of a cell) in the dataset, "
+ "Number, A numerical value from a cell in the dataset, which may represent a computed statistic (e.g., average, maximum, minimum), "
+ "List[category], A list containing a fixed number of categories. The expected format is: '['cat', 'dog']'. Pay attention to the wording of the question to determine if uniqueness is required or if repeated values are allowed."
+ "List[number], Similar to List[category], but with numbers as its elements."
+ "Requirements: Only respond with the JSON, do not add any additional text only use respond in the JSON format."
)
def response_test(question:str, context:str, model:str = "gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": MESSAGE_SYSTEM_CONTENT,
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": context},
        ],
    )
    
    return response.choices[0].message.content

def run_question_test(query: str, eval_df:str):
    response = response_test(query, eval_df['DataRaw'])    
    return response
