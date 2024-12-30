from pytube import YouTube
from pydub import AudioSegment

# Function to download YouTube video and convert to audio


def download_youtube_audio(url, output_path):
    # Download the video
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    downloaded_file = video.download(output_path=output_path)

    # Convert to audio
    audio = AudioSegment.from_file(downloaded_file)
    audio_file = downloaded_file.replace('.mp4', '.mp3')
    audio.export(audio_file, format="mp3")

    return audio_file


# Example usage
url = 'https://youtu.be/F9n6ayKmlxc?si=kcZzmyFQZCYIX4bF'
output_path = './'
audio_file = download_youtube_audio(url, output_path)
print(f'Audio file saved at: {audio_file}')
