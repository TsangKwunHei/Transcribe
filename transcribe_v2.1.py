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


SUBJECT_KEYWORDS = ["ai", 
                    "biology",
                    "reading note"]


load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


# Audio Transcription Function

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


# Text Processing Functions

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
    """
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
        if any(re.match(pattern, line.strip(), re.IGNORECASE) for pattern in meta_patterns):
            continue
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

def process_with_gpt(text, client):
    """
    Process text through GPT to improve clarity while maintaining core meaning.
    """
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

            line = filler_pattern.sub("", line)
            line = re.sub(r"\s+", " ", line).strip()

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
        
        if client and cleaned_block:
            cleaned_block = process_with_gpt(cleaned_block, client)
            
        cleaned_paragraphs.append(cleaned_block)
    
    return cleaned_paragraphs


# Audio Recording Function

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

# Subject Determination Function

def determine_subject(raw_text, keywords):
    """
    Determines the subject based on the first or last five words of the raw transcription.
    Returns the subject keyword if found, otherwise returns None.
    """
    words = raw_text.split()
    if not words:
        return None
    first_five = " ".join(words[:5]).lower()
    last_five = " ".join(words[-5:]).lower()
    for keyword in keywords:
        if keyword.lower() in first_five or keyword.lower() in last_five:
            return keyword.lower()
    return None


# File Appending Functions

def append_to_weekly_file(text):
    """
    Append the transcription to a weekly file.
    File name format: w{week}_{Month}_{Year}.txt
    The latest transcription is at the top with a date header.
    """
    current_date = datetime.now()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "transcriptions")
    os.makedirs(base_dir, exist_ok=True)
    
    iso_calendar = current_date.isocalendar()
    week_number = iso_calendar[1]
    year = iso_calendar[0]
    month_str = current_date.strftime("%B")
    week_file = f"w{week_number}_{month_str}_{year}.txt"
    file_path = os.path.join(base_dir, week_file)

    current_date_str = current_date.strftime("%A, %B %d, %Y")
    header_line = f"[{current_date_str}]"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) > 2:
                existing_lines = lines[2:]
            else:
                existing_lines = []
    except FileNotFoundError:
        existing_lines = []

    filtered_lines = [line for line in existing_lines if line.strip() != header_line]
    existing_content = "".join(filtered_lines)

    new_block = f"{header_line}\n{text}\n\n"
    final_text = new_block + existing_content

    total_words = len(final_text.split())
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Total words: {total_words}\n\n{final_text}")

    return file_path

def append_to_subject_file(text, subject):
    """
    Append the transcription to a subject-specific file.
    File name format: {subject}.txt
    The latest transcription block includes a month header with a counter for transcriptions.
    If a transcription for the current month already exists, its header is removed and replaced.
    """
    current_date = datetime.now()
    current_month = current_date.strftime("%B")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "transcriptions")
    os.makedirs(base_dir, exist_ok=True)
    
    subject_file = os.path.join(base_dir, f"{subject.lower()}.txt")
    
    if os.path.exists(subject_file):
        with open(subject_file, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = ""
    
    lines = content.splitlines()
    count = 0
    if lines and re.match(r'^\[Month: (\w+) \| Count: (\d+)\]$', lines[0]):
        m = re.match(r'^\[Month: (\w+) \| Count: (\d+)\]$', lines[0])
        header_month = m.group(1)
        header_count = int(m.group(2))
        if header_month.lower() == current_month.lower():
            count = header_count
            content = "\n".join(lines[1:]).lstrip()
    
    count += 1
    new_header = f"[Month: {current_month} | Count: {count}]"
    new_block = f"{new_header}\n{text}\n\n"
    final_content = new_block + content
    
    with open(subject_file, "w", encoding="utf-8") as f:
        f.write(final_content)
    return subject_file

# ----------------------------------------------------------------------------
# Main Entry Point
# ----------------------------------------------------------------------------
def main():
    # Part 1: Record and Transcribe Audio
    audio_file_path = record_audio()
    raw_transcribed_text = whisper_transcribe(audio_file_path)
    
    if not raw_transcribed_text:
        print("Transcription failed or returned empty.")
        return

    # Part 2: Determine if transcription matches a subject based on criteria
    subject = determine_subject(raw_transcribed_text, SUBJECT_KEYWORDS)

    # Part 3: Formatting the Transcribed Text
    sentences = split_into_sentences(raw_transcribed_text)
    paragraphs = group_sentences_into_paragraphs(sentences, paragraph_size=3)
    uncleaned_output = "\n\n".join(paragraphs)
    
    # Part 4: Cleaning with GPT processing
    cleaned_paragraphs = basic_cleaning(paragraphs, client)
    final_output = "\n\n".join(cleaned_paragraphs)
    
    # Part 5: Append to the appropriate file based on subject criteria
    if subject:
        output_file = append_to_subject_file(final_output, subject)
        print(f"\nTranscription appended to subject-specific file: {output_file}")
    else:
        output_file = append_to_weekly_file(final_output)
        print(f"\nTranscription appended to weekly file: {output_file}")
    
    print("\nCleaned transcription (post-clean):")
    print(final_output)
    return final_output

if __name__ == "__main__":
    main()
