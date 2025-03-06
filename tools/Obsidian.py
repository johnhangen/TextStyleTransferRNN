#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import requests
import os 
import json
from dotenv import load_dotenv
from datetime import date
import urllib3
    
load_dotenv()

# running locally
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ObsidianCrawler:

    def __init__(self, url:str) -> None:
        self.url: str = url
        self.API_Key: str = os.getenv("ObsidianAPI")
        self.headers = {"Authorization": f"Bearer {self.API_Key}"}
        self._ignore = ['Files', 'Ink', 'Work']
        self.data = []

    def extract(self, query:str = "") -> None:
        if query == "":
            query = r"%20%20"

        url_post:str = self.url + f"search/simple/?query={query}&contextLength=100"
        logging.info(f"Getting all obsidian notes from {url_post}")

        json_data = json.loads(requests.post(url_post, headers=self.headers, verify=False).text)

        for _, note_dict in enumerate(json_data):
            note_name = note_dict["filename"].replace(" ", r"%20")
            if note_name.split("/")[0] not in self._ignore:

                note_json = requests.get(
                    f"https://127.0.0.1:27124/vault/{note_name}",
                    headers=self.headers,
                    verify=False
                    ).text
                
                self.data.append(note_json)
    
    def save(self, path:str = "") -> None:
        if path == "":
            path = f"data/obsidian_notes_{date.today()}.text"
        
        with open(path, "w") as file:
            json.dump("".join(self.data), file)


if __name__ == "__main__":
    obsidian_url = ObsidianCrawler('http://127.0.0.1:27123/')
    obsidian_url.extract()
    obsidian_url.save()