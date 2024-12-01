from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

class VoiceProcessor:
    def __init__(self):
        # Setup OpenAI API
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables")
        self.client = OpenAI(api_key=self.api_key)
        
        # Setup logging
        logging.basicConfig(
            filename=f'voice_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def text_to_speech(self, text):
        """
        Convert text to speech using OpenAI's TTS API
        """
        try:
            # First, convert to natural speech
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "Convert the input text into natural, conversational speech while preserving its meaning."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.7,
                max_tokens=150
            )

            natural_text = response.choices[0].message.content.strip()
            
            # Log the text conversion
            self.logger.info(f"Input: {text}")
            self.logger.info(f"Natural text: {natural_text}")
            print(f"\nProcessed text: {natural_text}")

            # Convert to speech
            speech_file_path = f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            
            # Use with_streaming_response for the audio generation
            with self.client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="echo",  # Using nova voice for clear and friendly tone
                input=natural_text
            ) as response:
                # Save the audio file
                with open(speech_file_path, 'wb') as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
            
            print(f"Audio saved to: {speech_file_path}")
            return speech_file_path, natural_text

        except Exception as e:
            self.logger.error(f"Error converting to speech: {str(e)}")
            print(f"Error: {str(e)}")
            return None, None

def main():
    processor = VoiceProcessor()
    
    # Example of receiving text from another model
    example_text = "Hello my name John I student here"
    
    print("Converting to speech...")
    speech_file, converted_text = processor.text_to_speech(example_text)
    
    if speech_file and converted_text:
        print("\nConversion successful!")
        print(f"Input text: {example_text}")
        print(f"Converted text: {converted_text}")
        print(f"Audio file: {speech_file}")
    else:
        print("\nConversion failed. Check the logs for details.")

if __name__ == "__main__":
    main()