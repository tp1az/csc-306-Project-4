Our implementation of Semeval 2025 Task 8
Important Files: llm_qa.py

To run our submission, first you import llm_qa.py file.
Then you create a run_LLM object, variable_name = llm_qa.run_LLM,
this class has 4 Parameters, api_key, model, prompt, and data size.
- The api_key is the only required parameter, and that requires the openai API key as a string. 
- The model is the name of the openAI model that is desired to be used, default is "gpt-4o-mini".
- The prompt is the type of prompting you would like to test, Right now we have Zero shot in context learning, which is the default, and Chain of thought.
  - Zero shot in context learning can be called by z-icl, zero-shot, or zero-shot-in-context-learning,
  - Chain of thought can be called by chain_of_thought, cot, or chain
- The dataSize is asking whether you would like to run the model with the All data, or the Sample data. Currently Sample is only working due to token overflow issues.
- Look at run.ipynb for further details on how to run this program
