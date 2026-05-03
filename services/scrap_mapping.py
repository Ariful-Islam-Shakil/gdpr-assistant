import os, json, resource
import requests
import re
class ScrapMapping:
    def __init__(self):
        pass
    def extract_recitals_with_number(self, text):
        """
        Extracts all GDPR recital URLs and their numbers from the text.

        Args:
            text (str): Web scraped content.

        Returns:
            List[dict]: List of dictionaries containing 'number' and 'url'.
        """
        pattern = r"(https:\/\/gdpr-info\.eu\/recitals\/no-([1-9]|[1-9][0-9]|1[0-6][0-9]|17[0-3])\/)"
        regex = re.compile(pattern)

        results = []
        for line in text.splitlines():
            line = line.strip()
            for match in regex.findall(line):
                url = match[0]
                number = match[1]
                # results.append({"number": number, "url": url})
                results.append(number)

        return results

    def save_scrap_mapping(self,maping):
        map_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "scrap_mapping.json")
        with open(map_path, "w") as f:
            json.dump(maping, f, indent=4)

    def get_recitals_number(self):
        maping = []
        for i in range(1,99,1):
            url = f"https://gdpr-info.eu/art-{i}-gdpr/"

            response = requests.get(url)
            print(type(response.text))
            result = self.extract_recitals_with_number(response.text)
            maping.append({"article": i, "recitals": result})
            if i%10 == 0:
                print(f"\nScraped {i} articles\n")
        self.save_scrap_mapping(maping)

        return maping

if __name__ == "__main__":
    scrap_mapping = ScrapMapping()
    scrap_mapping.get_recitals_number()