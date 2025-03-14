Our implementation of Semeval 2025 Task 8
Important Files: llm_qa.p, fine_tuning.py, evaluation.ipynb

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

To run fine_tuning simply run the file and give input when prompted. To evaluate this model note 
that you need to change the evaluation strategy in TrainingArguments. Note: to run this make sure all dependencies are installed.

To run evaluation.ipynb, simply look through the notebook until you see the context generation functions. From 
there look at the actual qa querying functions. In order to evaluate a combination of context generation strategy, 
and qa querying strategy, you run the cell under the label "Get evaluation data from model". To evaluate different
strategies simply change the line runs the llm (more directions in notebook). From there simply format the outputted data, 
and run the last cell. The formatting functions are also given.