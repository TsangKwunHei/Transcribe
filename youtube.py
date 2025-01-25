import yt_dlp
import os
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

def split_on_silence(audio_path, min_silence_len=1000, silence_thresh=-40, chunk_length_ms=180000):
    """
    Split audio file into chunks at silence points
    chunk_length_ms = 3 minutes (180000ms)
    min_silence_len = 1 second (1000ms)
    """
    audio = AudioSegment.from_wav(audio_path)
    
    # Initialize chunks
    chunks = []
    start_time = 0
    
    while start_time < len(audio):
        # Get a chunk of desired length
        end_time = min(start_time + chunk_length_ms, len(audio))
        chunk = audio[start_time:end_time]
        
        if end_time < len(audio):
            # Find the nearest silence point after chunk_length
            silence_points = detect_nonsilent(chunk, min_silence_len, silence_thresh)
            if silence_points:
                # Adjust end_time to the last non-silent segment
                end_time = start_time + silence_points[-1][1]
        
        chunks.append(audio[start_time:end_time])
        start_time = end_time
    
    return chunks

def download_youtube_audio(url, output_path="downloads"):
    """Download audio from YouTube video"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Sanitize output filename
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        # Use video ID instead of title for the filename
        'outtmpl': os.path.join(output_path, '%(id)s.%(ext)s'),
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Use video ID for the audio file name
            audio_file = os.path.join(output_path, f"{info['id']}.wav")
            return audio_file, info['title']
    except Exception as e:
        print(f"Error downloading YouTube video: {str(e)}")
        return None, None

def transcribe_chunk(chunk, chunk_path):
    """Transcribe a single audio chunk"""
    try:
        # Export chunk to temporary file
        chunk.export(chunk_path, format="wav")
        
        with open(chunk_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        print(f"Error transcribing chunk: {str(e)}")
        return None
    finally:
        if os.path.exists(chunk_path):
            os.remove(chunk_path)

def main():
    # YouTube video URL
    video_url = input("Enter YouTube video URL: ")
    
    # Download audio
    print("Downloading audio...")
    audio_file, video_title = download_youtube_audio(video_url)
    
    if audio_file:
        print("Audio downloaded successfully!")
        print("Splitting audio into chunks...")
        
        chunks = split_on_silence(audio_file)
        print(f"Split into {len(chunks)} chunks")
        
        # Process each chunk
        all_transcriptions = []
        for i, chunk in enumerate(chunks, 1):
            print(f"\nProcessing chunk {i}/{len(chunks)}...")
            chunk_path = f"downloads/temp_chunk_{i}.wav"
            
            transcription = transcribe_chunk(chunk, chunk_path)
            if transcription:
                all_transcriptions.append(transcription)
        
        # Combine all transcriptions
        final_transcription = " ".join(all_transcriptions)
        
        # Save complete transcription
        output_file = os.path.join("downloads", f"{video_title}_transcription.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_transcription)
        
        print(f"\nComplete transcription saved to: {output_file}")
        
        # Clean up original audio file
        os.remove(audio_file)
        print("Audio file cleaned up")

if __name__ == "__main__":
    main()
