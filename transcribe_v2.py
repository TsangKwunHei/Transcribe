import os
import subprocess
import re
import wave
import pyaudio
import sys
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------------------------------------------------------
# Load environment variables and initialize OpenAI client
# ----------------------------------------------------------------------------
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ----------------------------------------------------------------------------
# Audio Transcription Function
# ----------------------------------------------------------------------------
def whisper_transcribe(audio_file_path):
    """
    Transcribe the audio file using OpenAI's Whisper API.
    If the file exceeds 25MB, compress it using ffmpeg before transcribing.
    """
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
    
    try:
        file_size = os.path.getsize(audio_file_path)
        if file_size > MAX_FILE_SIZE:
            print(f"Warning: File size ({file_size/1024/1024:.1f}MB) exceeds OpenAI's 25MB limit.")
            print("Compressing audio file...")
            
            # Create a temporary filename for the compressed file
            temp_filename = audio_file_path.replace(".wav", "_compressed.wav")
            
            # Use ffmpeg to compress the file
            command = f'ffmpeg -i "{audio_file_path}" -ar 16000 -ac 1 -b:a 48k "{temp_filename}"'
            subprocess.run(command, shell=True, check=True)
            
            # Use the compressed file for transcription
            audio_file_path = temp_filename

        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        # Clean up temporary file if it exists
        if audio_file_path.endswith("_compressed.wav"):
            os.remove(audio_file_path)
        
        return transcript.text

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return None

# ----------------------------------------------------------------------------
# Text Processing
# ----------------------------------------------------------------------------
def split_into_sentences(text):
    """
    Improved sentence splitter using regex that separates on .?! but 
    keeps punctuation with the preceding token.
    """
    text = text.strip()
    parts = re.split(r'([.?!])', text)
    sentences = []
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        if i + 1 < len(parts):
            punctuation = parts[i + 1]
            chunk += punctuation
        if chunk:
            sentences.append(chunk.strip())
    return sentences

def group_sentences_into_paragraphs(sentences, paragraph_size=3):
    """
    Group sentences into paragraphs based on a fixed number of sentences.
    """
    paragraphs = []
    for i in range(0, len(sentences), paragraph_size):
        group = sentences[i : i + paragraph_size]
        paragraph_text = " ".join(group)
        paragraphs.append(paragraph_text)
    return paragraphs

def remove_meta_talk(text):
    """
    Removes common lines that might appear in GPT responses, such as
    'Certainly!', 'Sure!', or 'As an AI language model...' 
    This is a simple regex-based filter.
    """
    # Define some patterns that typically indicate "meta" lines
    meta_patterns = [
        r"^Certainly.*",
        r"^Sure.*",
        r"^As an AI.*",
        r"^I'm an AI.*",
        r"^Here.*text.*",
        r"^Of course.*"
    ]
    
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        # If line matches any "meta" pattern (case-insensitive), skip it
        if any(re.match(pattern, line.strip(), re.IGNORECASE) for pattern in meta_patterns):
            continue
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

def process_with_gpt(text, client):
    """
    Process text through GPT to improve clarity while maintaining core meaning.
    """
    # Enhanced system prompt that explicitly forbids meta disclaimers
    system_prompt = (
        "You are a minimal text editor. Your task is to:\n"
        "1. Fix only obvious grammar mistakes\n"
        "2. Remove clear redundancies and filler words\n"
        "3. Make only necessary structural improvements\n"
        "4. Preserve the original tone and speaking style\n"
        "5. Keep all specific details and examples exactly as given\n"
        "6. Do NOT include any meta text like 'Certainly!' or disclaimers. "
        "   Your output should only be the cleaned-up user text.\n\n"
        "Important: Make minimal changes. If a sentence is understandable, leave it as is.\n"
        "Never paraphrase or rewrite unless absolutely necessary for clarity."
    )

    user_prompt = (
        f"Please make minimal improvements to this transcribed speech, "
        f"focusing only on essential grammar fixes and removal of filler words:\n\n{text}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or your preferred model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        gpt_output = response.choices[0].message.content.strip()
        # Remove any possible meta lines
        cleaned_output = remove_meta_talk(gpt_output)
        return cleaned_output
    except Exception as e:
        print(f"Error processing text with GPT: {str(e)}")
        return text  # Return original text if processing fails

def basic_cleaning(final_paragraph_blocks, client=None):
    """
    Basic cleaning with optional GPT processing:
    1. Remove filler words (um, uh, etc.)
    2. Capitalize first character of each sentence
    3. If client is provided, optionally process through GPT
    """
    filler_pattern = re.compile(r"\b(um|uh|like|you know|i mean)\b", re.IGNORECASE)
    cleaned_paragraphs = []

    for block in final_paragraph_blocks:
        lines = block.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove filler words
            line = filler_pattern.sub("", line)
            # Remove extra spaces
            line = re.sub(r"\s+", " ", line).strip()

            # Capitalize first character if not bullet
            if line.startswith("* "):
                bullet_content = line[2:].strip()
                if bullet_content:
                    bullet_content = bullet_content[0].upper() + bullet_content[1:]
                cleaned_lines.append("* " + bullet_content)
            else:
                if line:
                    line = line[0].upper() + line[1:]
                cleaned_lines.append(line)
        
        cleaned_block = "\n".join(cleaned_lines)
        
        # If OpenAI client is provided, do a GPT pass
        if client and cleaned_block:
            cleaned_block = process_with_gpt(cleaned_block, client)
            
        cleaned_paragraphs.append(cleaned_block)
    
    return cleaned_paragraphs

# ----------------------------------------------------------------------------
# Audio Recording and File Appending
# ----------------------------------------------------------------------------
def record_audio(output_filename="recorded_audio.wav"):
    """
    Record audio from the microphone until Enter is pressed.
    """
    audio_format = pyaudio.paInt16
    channels = 1
    sample_rate = 44100
    chunk_size = 1024

    audio = pyaudio.PyAudio()
    stream = audio.open(format=audio_format, 
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    print("Recording... Press Enter to stop.")
    frames = []

    try:
        while True:
            data = stream.read(chunk_size)
            frames.append(data)
            # Check if Enter is pressed
            if os.name == 'nt':  # Windows
                import msvcrt
                if msvcrt.kbhit() and msvcrt.getch() == b'\r':
                    break
            else:  # Mac/Linux
                import select
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    break
    except KeyboardInterrupt:
        pass

    print("Recording stopped.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(output_filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(audio_format))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    return output_filename

def append_to_text_file(text):
    """
    Always store the file in a 'transcriptions' folder relative to this script.
    The file is named w{week}_{Month}_{Year}.txt.
    This updated version ensures that for transcriptions on the same day,
    only the latest transcription has the date header.
    """
    current_date = datetime.now()
    
    # Path is always relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "transcriptions")
    os.makedirs(base_dir, exist_ok=True)
    
    # File name: wWeek_Month_Year.txt
    iso_calendar = current_date.isocalendar()
    week_number = iso_calendar[1]
    year = iso_calendar[0]
    month_str = current_date.strftime("%B")
    week_file = f"w{week_number}_{month_str}_{year}.txt"
    file_path = os.path.join(base_dir, week_file)

    # Today's date header line
    current_date_str = current_date.strftime("%A, %B %d, %Y")
    header_line = f"[{current_date_str}]"

    # Read existing content (skip the first two header lines if present)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Assume the first two lines are the "Total words:" header and a blank line
            if len(lines) > 2:
                existing_lines = lines[2:]
            else:
                existing_lines = []
    except FileNotFoundError:
        existing_lines = []

    # Remove any occurrence of today's header from the existing transcription blocks
    filtered_lines = [line for line in existing_lines if line.strip() != header_line]
    existing_content = "".join(filtered_lines)

    # Create new block: header + transcription text + some spacing
    new_block = f"{header_line}\n{text}\n\n"
    final_text = new_block + existing_content

    # Count total words and update the file header accordingly
    total_words = len(final_text.split())
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Total words: {total_words}\n\n{final_text}")

    return file_path
# ----------------------------------------------------------------------------
# Main Entry Point
# ----------------------------------------------------------------------------
def main():
    # Part 1: Record and Transcribe Audio
    audio_file_path = record_audio()
    transcribed_text = whisper_transcribe(audio_file_path)
    
    if not transcribed_text:
        print("Transcription failed or returned empty.")
        return

    # Part 2: Formatting the Transcribed Text
    sentences = split_into_sentences(transcribed_text)
    paragraphs = group_sentences_into_paragraphs(sentences, paragraph_size=3)
    uncleaned_output = "\n\n".join(paragraphs)
    
    # Part 3: Cleaning with GPT processing
    cleaned_paragraphs = basic_cleaning(paragraphs, client)
    final_output = "\n\n".join(cleaned_paragraphs)
    
    # Part 4: Append to file and print results
    output_file = append_to_text_file(final_output)
    print("\nCleaned transcription (post-clean):")
    print(final_output)
    print(f"\nSuccessfully appended to {output_file}")

    return final_output

if __name__ == "__main__":
    main()
