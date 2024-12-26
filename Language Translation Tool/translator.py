import requests
import time

class LanguageTranslator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.max_retries = 5 
        
        self.languages = {
            "af": "Afrikaans",
            "ar": "Arabic",
            'az': "Azerbaijani",
            'bg': "Bulgarian",
            "ca": "Catalan",
            "cs": "Czech",
            "cy": "Welsh",
            "da": "Danish",
            "de": "German",
            "el": "Greek",
            "en": "English",
            "eo": "Esperanto",
            "es": "Spanish",
            "et": "Estonian",
            "fi": "Finnish",
            "fr": "French",
            "ga": "Irish",
            "gl": "Galician",
            "he": "Hebrew",
            "hi": "Hindi",
            "hu": "Hungarian",
            "hy": "Armenian",
            "id": "Indonesian",
            'is': "Icelandic",
            'it': 'Italian',
            'mk': 'Macedonian',
            'ml': 'Malayalam',
            'mr': 'Marathi',
            'mt': 'Maltese',
            'nl': 'Dutch',
            'ro': 'Romanian',
            'ru': 'Russian',
            'sk': 'Slovak',
            'sq': 'Albanian',
            'sv': 'Swedish',
            'sw': 'Swahili',
            'tl': 'Tagalog',
            'uk': 'Ukrainian',
            'ur': 'Urdu',
            'vi': 'Vietnamese',
            'zh': 'Chinese'
            }

    def get_supported_languages(self):
        print("\nSupported Languages:")
        print("-------------------")
        for code, language in self.languages.items():
            print(f"{code}: {language}")

    def wait_for_model(self, response) -> tuple[bool, float]:
        try:
            error_data = response.json()
            if "error" in error_data and "estimated_time" in error_data:
                if "is currently loading" in error_data["error"]:
                    return True, error_data["estimated_time"]
        except:
            pass
        return False, 0

    def translate(self, text: str, target_lang: str, source_lang: str = "en") -> dict:
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                model_url = f"{self.api_url}-{source_lang}-{target_lang}"
                
                response = requests.post(
                    model_url,
                    headers=self.headers,
                    json={"inputs": text}
                )
                
                is_loading, wait_time = self.wait_for_model(response)
                
                if is_loading:
                    wait_time = min(wait_time, 30)
                    print(f"\nModel is loading... Waiting {wait_time:.1f} seconds (Attempt {retry_count + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        translated_text = result[0].get('translation_text', '')
                        if translated_text:
                            return {
                                "translated_text": translated_text,
                                "source_lang": source_lang,
                                "target_lang": target_lang
                            }
                
                return {"error": f"API request failed: {response.text}"}

            except Exception as e:
                return {"error": f"Translation failed: {str(e)}"}
            
        return {"error": "Maximum retries reached. Model failed to load."}

def main():
    print("You need a Hugging Face API key to use this translator.")
    print("Get one from: https://huggingface.co/settings/tokens")
    api_key = input("Please enter your Hugging Face API key: ").strip()
    
    if not api_key:
        print("Error: API key is required.")
        return

    translator = LanguageTranslator(api_key)
    
    while True:
        print("\n=== Language Translator ===")
        print("1. View supported languages")
        print("2. Translate text")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            translator.get_supported_languages()
            
        elif choice == "2":
            translator.get_supported_languages()
            source_lang = input("\nEnter source language code (or press Enter for English): ").lower()
            if source_lang == "":
                source_lang = "en"
                
            if source_lang not in translator.languages:
                print(f"Error: '{source_lang}' is not a supported language code.")
                continue
                
            target_lang = input("Enter target language code: ").lower()
            if target_lang not in translator.languages:
                print(f"Error: '{target_lang}' is not a supported language code.")
                continue
                
            text = input("\nEnter text to translate: ")
            if not text.strip():
                print("Error: Please enter some text to translate.")
                continue
                
            print("\nTranslating...")
            result = translator.translate(text, target_lang, source_lang)
            
            if "error" in result:
                print(f"\nError: {result['error']}")
            else:
                print("\nResults:")
                print(f"Original ({translator.languages[source_lang]}): {text}")
                print(f"Translated ({translator.languages[target_lang]}): {result['translated_text']}")
                
        elif choice == "3":
            print("\nThank you for using the translator. Goodbye!")
            break
            
        else:
            print("\nInvalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()