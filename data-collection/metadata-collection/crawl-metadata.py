import json
import time
import csv
from datetime import datetime
import requests
import yaml


class HuggingFaceAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = 'https://huggingface.co'
        self.interval = 1  # req interval 

    def set_interval(self, interval):
        self.interval = interval

    # Library name
    # Tasks in buckets (pipeline_tag): NLP, CV, Audio,  
    # Min likes: 2
    # Min age since upload: 6 months 
    # Min commits: TBD 


    def get_all_models(self, max_results=1):
        models = []
        next_page_url = f"/api/models?limit={max_results}"
        total_count = 0

        while next_page_url:
            print(f"Processing URL: {next_page_url}")
            response = requests.get(self.base_url + next_page_url)
            response_dict = json.loads(response.content)

            if not response_dict:
                break

            models.extend(response_dict)
            total_count += len(response_dict)
            print(f"Have obtained {total_count} Models")
###############################
            current_date = datetime.now().strftime("%Y-%m-%d")
            file_count = len(models)
        
            json_file_name = f'output/{current_date}_all_huggingface_models_{file_count}.json'
            csv_file_name = f'output/{current_date}_all_huggingface_models_{file_count}.csv'
        
            # saving JSON
            with open(json_file_name, 'w', encoding='utf-8') as f:
                json.dump(models, f, indent=4)
            print(f"List of models saved in JSON file: {json_file_name}")
        
            # saving CSV
            json_to_csv(json_file_name, csv_file_name)
            print(f"List of models saved in CSV files: {csv_file_name}")
#############################
            next_link = response.links.get("next", {})
            next_page_url = next_link.get("url", "").replace(self.base_url, "") if next_link else None

            time.sleep(self.interval)

        return models


def json_to_csv(json_file, csv_file):
    with open(json_file, 'r', encoding='utf-8') as jf:
        data = json.load(jf)

    with open(csv_file, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['Model ID', 'Other Details'])

        for model in data:
            model_id = model.get('modelId', '')
            other_details = json.dumps(model, ensure_ascii=False)
            writer.writerow([model_id, other_details])


if __name__ == '__main__':
    # with open('data-collection/metadata-collection/config.yaml', 'r') as f:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    api_key = config.get("huggingface_key")
    hf_api = HuggingFaceAPI(api_key)

    all_models = hf_api.get_all_models()
    # current_date = datetime.now().strftime("%Y-%m-%d")
    # file_count = len(all_models)

    # # rel path saved:  metadata-collection
    # output_dir = 'data-collection/metadata-collection'
    # json_file_name = f'{output_dir}/{current_date}_all_huggingface_models_{file_count}.json'
    # csv_file_name = f'{output_dir}/{current_date}_all_huggingface_models_{file_count}.csv'

    # # saving JSON
    # with open(json_file_name, 'w', encoding='utf-8') as f:
    #     json.dump(all_models, f, indent=4)
    # print(f"List of models saved in JSON file: {json_file_name}")

    # # saving CSV
    # json_to_csv(json_file_name, csv_file_name)
    # print(f"List of models saved in CSV files: {csv_file_name}")

