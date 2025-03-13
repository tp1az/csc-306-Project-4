import pandas as pd
import ast
from datasets import Dataset
from databench_eval import Runner, Evaluator
from openai import OpenAI
import numpy as np
import tiktoken

class Loader:
    def  __init__(self, dataSize = 'sample'):
        self.dataSize = dataSize 
    
    def load_qa(self):
        test_qa = pd.read_csv("~/csc-306-Project-4/src/data/test_qa.csv")  
        test_qa = test_qa.set_index('dataset')
        return test_qa
    
    def get_dataSize(self):
        return self.dataSize
    
    def load_dataset(self):
        ibm_hr = pd.read_parquet(f"~/csc-306-Project-4/src/data/066_IBM_HR/{self.dataSize}.parquet")
        tripadvisor = pd.read_parquet(f"~/csc-306-Project-4/src/data/067_TripAdvisor/{self.dataSize}.parquet")
        worldBank = pd.read_parquet(f"~/csc-306-Project-4/src/data/068_WorldBank_Awards/{self.dataSize}.parquet")
        taxonomy = pd.read_parquet(f"~/csc-306-Project-4/src/data/069_Taxonomy/{self.dataSize}.parquet")
        openfoodfacts = pd.read_parquet(f"~/csc-306-Project-4/src/data/070_OpenFoodFacts/{self.dataSize}.parquet")
        col = pd.read_parquet(f"~/csc-306-Project-4/src/data/071_COL/{self.dataSize}.parquet")
        admissions = pd.read_parquet(f"~/csc-306-Project-4/src/data/072_Admissions/{self.dataSize}.parquet")
        med_cost = pd.read_parquet(f"~/csc-306-Project-4/src/data/073_Med_Cost/{self.dataSize}.parquet")
        lift = pd.read_parquet(f"~/csc-306-Project-4/src/data/074_Lift/{self.dataSize}.parquet")
        mortality = pd.read_parquet(f"~/csc-306-Project-4/src/data/075_Mortality/{self.dataSize}.parquet")
        nba = pd.read_parquet(f"~/csc-306-Project-4/src/data/076_NBA/{self.dataSize}.parquet")
        gestational = pd.read_parquet(f"~/csc-306-Project-4/src/data/077_Gestational/{self.dataSize}.parquet")
        fires = pd.read_parquet(f"~/csc-306-Project-4/src/data/078_Fires/{self.dataSize}.parquet")
        coffee = pd.read_parquet(f"~/csc-306-Project-4/src/data/079_Coffee/{self.dataSize}.parquet")
        books = pd.read_parquet(f"~/csc-306-Project-4/src/data/080_Books/{self.dataSize}.parquet")
        data = pd.DataFrame(columns = ['DataSet','DataRaw'])
        data.loc[len(data)] = ['066_IBM_HR', ibm_hr.to_csv(index=False)]
        data.loc[len(data)] = ['067_TripAdvisor', tripadvisor.to_csv(index=False)]
        data.loc[len(data)] = ['068_WorldBank_Awards', worldBank.to_csv(index=False)]
        data.loc[len(data)] = ['069_Taxonomy', taxonomy.to_csv(index=False)]
        data.loc[len(data)] = ['070_OpenFoodFacts', openfoodfacts.to_csv(index=False)]
        data.loc[len(data)] = ['071_COL', col.to_csv(index=False)]
        data.loc[len(data)] = ['072_Admissions', admissions.to_csv(index=False)]
        data.loc[len(data)] = ['073_Med_Cost', med_cost.to_csv(index=False)]
        data.loc[len(data)] = ['074_Lift', lift.to_csv(index=False)]
        data.loc[len(data)] = ['075_Mortality', mortality.to_csv(index=False)]
        data.loc[len(data)] = ['076_NBA', nba.to_csv(index=False)]
        data.loc[len(data)] = ['077_Gestational', gestational.to_csv(index=False)]
        data.loc[len(data)] = ['078_Fires', fires.to_csv(index=False)]
        data.loc[len(data)] = ['079_Coffee', coffee.to_csv(index=False)]
        data.loc[len(data)] = ['080_Books', books.to_csv(index=False)]
        data = data.set_index('DataSet')
        return data

class OpenAI_API_setup:
    def __init__(self,api_key, model='gpt-4o-mini', prompt='Z-ICL'):
        self.api_key = api_key
        self.client = OpenAI(api_key = self.api_key)
        self.model = model
        self.prompt = prompt.lower()

    def get_message(self, question:str, context:str):
        message_system_content = self.get_prompt()
        message = [
                {
                    "role": "system",
                    "content": message_system_content,
                },
                {"role": "user", "content": question},
                {"role": "assistant", "content": context},]
        return message
    
    def response(self, question:str, context:str):
        message_system_content = self.get_prompt()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": message_system_content,
                },
                {"role": "user", "content": question},
                {"role": "assistant", "content": context},
            ],
        )
        
        return response.choices[0].message.content
    
    def get_prompt(self):
        if self.prompt == 'z-icl' or self.prompt == 'zero-shot' or self.prompt == 'zero-shot-in-context-learning':
            return self.Z_ICL()
        elif self.prompt == 'chain_of_thought' or self.prompt == 'cot' or self.prompt == 'chain':
            return self.Chain_of_Thought()
        else:
            return self.Z_ICL()

    def Z_ICL(self):
        MESSAGE_SYSTEM_CONTENT = ("You are an assistant tasked with answering the questions asked of a given CSV in JSON format. Your response should follow this guidlines {'Answer': answer}"
+ "answer: answer using information from the provided CSV only. The answer should be one of these categories: "
+ "Boolean, Valid answers include True/False, Y/N, Yes/No (all case insensitive), "
+ "Category, A value from a cell (or a substring of a cell) in the dataset, "
+ "Number, A numerical value from a cell in the dataset, which may represent a computed statistic (e.g., average, maximum, minimum), "
+ "List[category], A list containing a fixed number of categories. The expected format is: '['cat', 'dog']'. Pay attention to the wording of the question to determine if uniqueness is required or if repeated values are allowed."
+ "List[number], Similar to List[category], but with numbers as its elements."
+ "Requirements: Only respond with the JSON, do not add any additional text only use respond in the JSON format. Respond with {Answer : your response}, if your response is a list do {Answer : list}"
)
        return MESSAGE_SYSTEM_CONTENT
    
    def Chain_of_Thought(self):
        chain_of_thought = ("given an example csv with employee data that has"
        + " columns[Employee_ID, Department, Job_Title, Salary, Years_Experience] when asked 'Does the employee with the highest salary have the most years of experience?"
        + " 1. Check the Salary column and find the employee_ID with the highest salary" 
        + " 2. Check the Years_Experience column and find the Employee_ID with the most years of experience" 
        + "3. Compare the two employee_ID's and if the employee_ID with the highest salary has the most years of experience respond with True else respond with False")
        MESSAGE_SYSTEM_CONTENT = self.Z_ICL() + chain_of_thought
        return MESSAGE_SYSTEM_CONTENT

    def Few_shot(self):
        Few_shot = ("")
    
    
    
class run_LLM:
    def __init__(self,api_key, model = 'gpt-4o-mini', prompt='Z-ICL', dataSize = "sample"):
        self.LLM = OpenAI_API_setup(api_key = api_key, model=model, prompt=prompt)
        self.dataSize = dataSize
        self.data = Loader(dataSize)
        self.test_qa = self.data.load_qa()
        self.dataframe = self.data.load_dataset()
        self.model = model
    '''ChatGPT code on how to calculate token counts '''    
    def count_tokens(self, text):
        """Counts the number of tokens in a given text based on the model."""
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(str(text)))

    def get_answers(self):
        tokens = 0
        runner = 0
        final = []
        for name, query in self.test_qa.itertuples():
            if(runner % 10 == 0):
                print(runner)
            datas = self.dataframe.loc[name]
            tokens += self.count_tokens(self.LLM.get_message(query,datas['DataRaw']))
            result = self.LLM.response(query,datas['DataRaw'])
            tokens += self.count_tokens(result)
            result = result.replace("{", "").replace("}", "")
            final.append(result)
            runner += 1
        print(tokens)
        return final


    def clean(self, final: list):
        ans_list = []
        for i in final:
            if(":" in i ):
                ans = i.split(":")[1]
                ans_list.append(ans.replace("\n", "").replace("'", "").replace("```","").replace('"',""))#.replace("",""))
            else:
                ans_list.append(ans.replace("\n", "").replace("'", "").replace("```","").replace('"',"").replace("json", ""))
        return ans_list

    def convert_string(self, s):
        """Convert a string to a list, boolean, or float if possible."""
        s = s.strip()
        if s.lower() == "true" or s.lower() == "yes" or s.lower() == "y":
            return True
        elif s.lower() == "false" or s.lower() == "no" or s.lower() == "n":
            return False
        try:
            int_val = int(s)
            if str(int_val) == s:  
                return int_val
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass

        
        if(len(s) != 0):
            list_val = []
            if s[0] == "[" and s[-1] == "]":
                for i in s.replace("[","").replace("]","").split(","):
                    list_val.append(self.convert_string(i))
                return list_val

        # If no conversion is possible, return the original string
        return s


    def reformat_answers(self, answers):
        valid = []
        clean_answers = self.clean(answers)
        for i in clean_answers:
            valid.append(self.convert_string(i))
        return valid
    '''answer loading file from databench_eval'''
    def load_answers(self):
        qa_df = pd.DataFrame()
        qaa = self.test_qa
        # Read the answers from the .txt files into separate lists
        with open("~/csc-306-Project-4/src/data/answers.txt", "r") as f:
            answers = f.read().splitlines()

        with open("~/csc-306-Project-4/src/data/answers_lite.txt", "r") as f:
            sample_answers = f.read().splitlines()

        with open("~/csc-306-Project-4/src/data/semantics.txt", "r") as f:
            semantics = f.read().splitlines()

        # Combine the lists into a DataFrame

        # Load the dataset column from the specified file
        qaa["answer"] = answers
        qaa["sample_answer"] = sample_answers
        qaa["type"] = semantics
        return qaa
    '''Comparison file from databench_eval'''    
    def default_compare(self, value, truth, semantic):
        STRIP_CHARS = "[]'\" "
        semantic = semantic.strip()
        valid_null_set = [None, "nan", "", " ", np.nan, "np.nan", "None"]

        if str(value).strip(STRIP_CHARS) in valid_null_set and str(truth).strip(STRIP_CHARS) in valid_null_set:
            return True
        if str(value).strip(STRIP_CHARS) in valid_null_set or str(truth).strip(STRIP_CHARS) in valid_null_set:
            return False

        if semantic == "boolean":
            valid_true_values = ['true', 'yes', 'y']
            valid_false_values = ['false', 'no', 'n']
            value_str = str(value).strip(STRIP_CHARS).lower()
            truth_str = str(truth).strip(STRIP_CHARS).lower()
            return (value_str in valid_true_values and truth_str in valid_true_values) or (value_str in valid_false_values and truth_str in valid_false_values)
        elif semantic == "category":
            value_str = str(value).strip(STRIP_CHARS)
            truth_str = str(truth).strip(STRIP_CHARS)
            if value_str == truth_str:
                return True

            try:
                value_date = pd.to_datetime(value_str).date()
                truth_date = pd.to_datetime(truth_str).date()
                return value_date == truth_date
            except (ValueError, TypeError):
                if not value_str and not truth_str:
                    return True
                return value_str == truth_str
        elif semantic == "number":
            try:
                value_cleaned = ''.join(char for char in str(value) if char.isdigit() or char in ['.', '-'])
                truth_cleaned = ''.join(char for char in str(truth) if char.isdigit() or char in ['.', '-'])
                return round(float(value_cleaned), 2) == round(float(truth_cleaned), 2)
            except:
                return False
        elif semantic == "list[category]":
            try:
                value_list = [item.strip(STRIP_CHARS) for item in str(value).strip('[]').split(',')]
                truth_list = [item.strip(STRIP_CHARS) for item in str(truth).strip('[]').split(',')]
                value_list = [
                    v if v not in valid_null_set else ""
                    for v in value_list
                ]
                truth_list = [
                    t if t not in valid_null_set else "" for t in truth_list
                ]
                if len(value_list) != len(truth_list):
                    return False

                # Attempt to parse each item as a date
                try:
                    value_dates = [pd.to_datetime(item).date() for item in value_list]
                    truth_dates = [pd.to_datetime(item).date() for item in truth_list]
                    return set(value_dates) == set(truth_dates)
                except (ValueError, TypeError):
                    # If parsing as dates fails, compare as strings
                    return set(value_list) == set(truth_list)
            except Exception as exc:
                return False
        elif semantic == "list[number]":
            try:
                value_list = sorted(float(''.join(c for c in v.strip() if c.isdigit() or c in ['.', '-'])) for v in str(value).strip('[]').split(',') if v.strip())
                truth_list = sorted(float(''.join(c for c in t.strip() if c.isdigit() or c in ['.', '-'])) for t in str(truth).strip('[]').split(',') if t.strip())

                value_list = [int(v * 100) / 100 for v in value_list]
                truth_list = [int(t * 100) / 100 for t in truth_list]

                if len(value_list) != len(truth_list):
                    return False
                
                return set(value_list) == set(truth_list)
            except Exception as exc:
                return False
        else:
            raise Exception(f"Semantic not supported: {semantic}")
        
    def percentage_data(self, myanswers, lite = True):
        num = [156, 0]
        boolean = [129, 0]
        lnum = [91, 0]
        category = [74, 0]
        lcate = [72,0]

        for name, query, ty, my_answers in myanswers.itertuples():
                ans = self.default_compare(my_answers, query, ty)
                if(ans == False):
                    match ty:
                        case "number":
                            num[1] += 1
                        case "boolean":
                            boolean[1] += 1
                        case "list[number]":
                            lnum[1] += 1
                        case "category": 
                            category[1] += 1    
                        case "list[category]":
                            lcate[1] += 1
        
        return num, boolean, lnum, category, lcate

                
    def get_percentage(self, valid, qa, lite = True):
        if(lite == False):
            myanswers = qa.drop(columns=['question','sample_answer'])
            myanswers['my_answers'] = valid
            return self.percentage_data(myanswers,lite)
        else:
            myanswers = qa.drop(columns=['question','answer'])
            myanswers['my_answers'] = valid
            return self.percentage_data(myanswers,lite)
        
    
    def run(self, percentage_data = False):
        answers = self.get_answers()
        valid = self.reformat_answers(answers)
        qa = self.load_answers()
        evaluator = Evaluator(qa=Dataset.from_pandas(qa))
        if self.dataSize == "all":
            lite = False
        else:
            lite = True
        
        if(percentage_data == True):
            num, boolean, lnum, category, lcate = self.get_percentage(valid, qa, lite)
            print('numbers:' + str(((num[0]-num[1])/num[0]) * 100) + "%")
            print("boolean:" + str((((boolean[0]-boolean[1])/boolean[0]) * 100)) + "%")
            print("list[number]:" + str(((lnum[0]-lnum[1])/lnum[0]) * 100) + "%")
            print("category:" + str((((category[0]-category[1])/category[0]) * 100)) + "%")
            print("list[category]:" + str(((lcate[0]-lcate[1])/lcate[0]) * 100) + "%")
             
        return evaluator.eval(valid, lite=lite)