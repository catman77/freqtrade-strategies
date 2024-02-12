import html
import json

import requests

#requests.urllib3.disable_warnings()

class LocalGPT:
    """ A class to interact with a locally hosted LLM using text-generation-web api. """
    def __init__(self):
        # For local streaming, the websockets are hosted without ssl - http://
        self.HOST = '127.0.0.1:5000'
        self.URI = f'http://{self.HOST}/v1/chat/completions'
        
        self.searching = False

        self.query_str = ""
        self.answer = ""

    def search(self, query: str):        
        self.searching = True
        formatted_query = query.replace('\n', '\\n').replace('\t', '\\t')
        self.query_str = formatted_query

        headers = {
            "Content-Type": "application/json"
        }

        history = []

        history.append({"role": "user", "content": self.query_str})
        
        data = {
            "mode": "chat",
            "messages": history,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95
        }

        response = requests.post(self.URI, headers=headers, json=data, verify=False)

        if response.status_code == 200:       
            result = response.json()['choices'][0]['message']['content']
            history.append({"role": "assistant", "content": result})

            if result != "":
                formatted_output = result.replace('\\n', '\n').replace('\\t', '\t')
                #print(formatted_output)
                return formatted_output
            else:
                self.searching = False
                return ""
        else:
            self.searching = False
            return f"Error: {response.status_code}"

