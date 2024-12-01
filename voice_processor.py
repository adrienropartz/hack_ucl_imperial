from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import pygame
import io

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
        Convert text to speech using OpenAI's TTS API and play it directly
        """
        try:
            # Use with_streaming_response for the audio generation
            with self.client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="echo",  # Using echo voice for clear and friendly tone
                input=text
            ) as response:
                # Initialize pygame mixer if not already done
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                
                # Create a temporary file in memory
                with io.BytesIO() as audio_buffer:
                    # Write the streaming response to the buffer
                    for chunk in response.iter_bytes():
                        audio_buffer.write(chunk)
                    
                    # Reset buffer position
                    audio_buffer.seek(0)
                    
                    # Load and play the audio
                    pygame.mixer.music.load(audio_buffer)
                    pygame.mixer.music.play()
                    
                    # Wait for the audio to finish playing
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                
                return True, text

        except Exception as e:
            self.logger.error(f"Error converting to speech: {str(e)}")
            print(f"Error: {str(e)}")
            return False, None

    def correct_sign_output(self, text):
        """
        Correct and improve the text output from the sign language model using GPT-4,
        focusing primarily on spelling corrections and common word substitutions
        """
        try:
            # Correct spelling errors and improve natural flow
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Updated to GPT-4o
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that primarily corrects spelling mistakes and common word substitutions, while maintaining the original meaning. Focus on fixing misspelled words and only make minimal grammatical adjustments when necessary. Be concise."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent corrections
                max_tokens=100
            )

            corrected_text = response.choices[0].message.content.strip()
            
            # Log the correction
            self.logger.info(f"Original model output: {text}")
            self.logger.info(f"Corrected output: {corrected_text}")
            
            return corrected_text

        except Exception as e:
            self.logger.error(f"Error correcting text: {str(e)}")
            print(f"Error: {str(e)}")
            return None

    def process_and_speak_sign(self, text):
        """
        Process sign language output and convert it to speech
        """
        try:
            # First correct the text
            corrected_text = self.correct_sign_output(text)
            if corrected_text:
                # Then convert to speech
                success, natural_text = self.text_to_speech(corrected_text)
                if success:
                    print(f"\nProcessing complete!")
                    print(f"Original text: {text}")
                    print(f"Corrected text: {corrected_text}")
                    return True
            return False

        except Exception as e:
            self.logger.error(f"Error in process_and_speak_sign: {str(e)}")
            print(f"Error: {str(e)}")
            return False

def main():
    processor = VoiceProcessor()
    
    # Extended example outputs from sign language model with spelling mistakes
    example_outputs = [
        # Common daily activities
        "I nede to go to the stoer",              # need, store
        "The wether is beautifull today",          # weather, beautiful
        "Can you help me with my homewrok",        # homework
        "She is reding a bok in the librery",      # reading, book, library
        
        # Food and dining
        "Im going to the resturant",               # I'm, restaurant
        "The caffe has good sandwitches",          # cafe, sandwiches
        "Let's get sum piza for diner",            # some, pizza, dinner
        "I want a hamberger and fris",             # hamburger, fries
        
        # Education
        "The techer explained the problm",         # teacher, problem
        "My mathmatiks test is tomorow",           # mathematics, tomorrow
        "I forgot my notbok at skool",             # notebook, school
        "The chemestry experimet failed",          # chemistry, experiment
        
        # Technology
        "The computr is not workin",               # computer, working
        "My phon batery is ded",                   # phone, battery, dead
        "I cant conect to the intrnet",            # can't, connect, internet
        "The sofware needs an updaet",             # software, update
        
        # Transportation
        "The buss is runing late",                 # bus, running
        "We mised the trane this mornin",          # missed, train, morning
        "The trafic is terible today",             # traffic, terrible
        "The taxe driver was verry nice",          # taxi, very
        
        # Work and office
        "The presntation starts at thre",          # presentation, three
        "Send the documnt by emale",               # document, email
        "The meting room is ocupied",              # meeting, occupied
        "I need to print these pappers",           # papers
        
        # Home and daily life
        "Please dont forget to lock the dor",      # don't, door
        "The dishwashr is broked",                 # dishwasher, broken
        "Turn off the lihgts befor leaving",       # lights, before
        "The airconditionr needs repare",          # air conditioner, repair
        
        # Social and communication
        "Thnak you for your hlep",                 # Thank, help
        "Can you repet that questin",              # repeat, question
        "Im wating in the cafetria",               # I'm, waiting, cafeteria
        "She didnt receve my mesage"               # didn't, receive, message
    ]
    
    print("Processing sign language outputs to speech...")
    for text in example_outputs:
        success = processor.process_and_speak_sign(text)
        if not success:
            print("\nProcessing failed. Check the logs for details.")

if __name__ == "__main__":
    main()