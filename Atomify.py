from langchain_openai import OpenAI
import pandas as pd
from langchain.prompts import PromptTemplate
import getpass
import os
import json

os.environ["OPENAI_API_KEY"] =""

llm = OpenAI(
    model="gt-3.5-turbo-instruct",
    temperature=0.2,
    max_retries=2,
)

def read_client_csv(csv_path):
    return pd.read_csv(csv_path)

STANDARD_SCHEMA = {
    "sku": "Unique SKU code",
    "quantity": "Number of items in stock",
    "warehouse_location": "the warehouse location"
   
}

def construct_prompt(df, standard_schema):
    sample_data = df.sample(n=5).to_dict(orient="records")

    # sample_data = df.head(5).to_dict(orient="records")
    sample_str = json.dumps(sample_data, indent=2)

    
    system_prompt = f"""
    You are a data transformation assistant. You are given a client's CSV data with unknown column description.
    You need to map or transform those columns to the following standardized inventory schema fields:

    {json.dumps(standard_schema, indent=2)}

    Return what you understand from the data and give a python code how to transfrom the csv to standard schema field to the likely column in the client's CSV.
    If a column is not found, set its value to None.
    Also provide any reasoning or transformations if necessary.
    """


    user_prompt = f"""
    Client CSV columns: {list(df.columns)}
    Sample rows:
    {sample_str}
    """

    return system_prompt, user_prompt

def main(csv_path):
    client_df = read_client_csv(csv_path)
    
    system_prompt, user_prompt = construct_prompt(client_df, STANDARD_SCHEMA)
    query = system_prompt + user_prompt
    print(query)
    response = llm.call(query)
    print(response) 
    

if __name__ == "__main__":
    main("csv_path")




