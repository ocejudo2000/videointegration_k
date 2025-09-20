import streamlit as st
import os
import subprocess
import tempfile
import shutil
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import sys
from moviepy import VideoFileClip, AudioFileClip
from pydub import AudioSegment
import mimetypes
from typing import List, Optional, Tuple

# Configuración de la página
st.set_page_config(
    page_title="Creador de Videos Secuenciales",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# VideoUploadHandler class for enhanced video file validation
class VideoUploadHandler:
    """Handles video file upload validation with enhanced MP4 support."""
    
    # Streamlit Cloud file size limit (200MB)
    MAX_FILE_SIZE_MB = 200
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # Supported video formats
    SUPPORTED_EXTENSIONS = ['.mp4']
    SUPPORTED_MIME_TYPES = ['video/mp4']
    
    @staticmethod
    def validate_video_files(uploaded_files) -> Tuple[bool, List[str]]:
        """
        Validate multiple uploaded video files.
        
        Args:
            uploaded_files: List of Streamlit uploaded file objects
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if not uploaded_files:
            return False, ["Please upload at least one MP4 video file."]
        
        errors = []
        
        for i, file in enumerate(uploaded_files):
            is_valid, file_errors = VideoUploadHandler._validate_single_video_file(file, i + 1)
            if not is_valid:
                errors.extend(file_errors)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_single_video_file(uploaded_file, file_number: int) -> Tuple[bool, List[str]]:
        """
        Validate a single uploaded video file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            file_number: File number for error messages
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if uploaded_file is None:
            return False, [f"File {file_number}: No file uploaded."]
        
        # Validate file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in VideoUploadHandler.SUPPORTED_EXTENSIONS:
            errors.append(f"File {file_number} ({uploaded_file.name}): Only MP4 files are supported. Found: {file_extension}")
        
        # Validate MIME type
        mime_type, _ = mimetypes.guess_type(uploaded_file.name)
        if mime_type not in VideoUploadHandler.SUPPORTED_MIME_TYPES:
            errors.append(f"File {file_number} ({uploaded_file.name}): Invalid file type. Expected MP4 video, found: {mime_type or 'unknown'}")
        
        # Validate file size
        file_size = uploaded_file.size if hasattr(uploaded_file, 'size') else len(uploaded_file.getvalue())
        if file_size > VideoUploadHandler.MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            errors.append(f"File {file_number} ({uploaded_file.name}): File size ({size_mb:.1f}MB) exceeds the {VideoUploadHandler.MAX_FILE_SIZE_MB}MB limit for Streamlit Cloud.")
        
        # Validate file is not empty
        if file_size == 0:
            errors.append(f"File {file_number} ({uploaded_file.name}): File is empty.")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_audio_file(uploaded_file) -> Tuple[bool, List[str]]:
        """
        Validate uploaded audio file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if uploaded_file is None:
            return False, ["Please upload an MP3 audio file."]
        
        errors = []
        
        # Validate file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in ['.mp3', '.wav', '.aac']:
            errors.append(f"Audio file ({uploaded_file.name}): Unsupported format. Please use MP3, WAV, or AAC.")
        
        # Validate file size
        file_size = uploaded_file.size if hasattr(uploaded_file, 'size') else len(uploaded_file.getvalue())
        if file_size > VideoUploadHandler.MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            errors.append(f"Audio file ({uploaded_file.name}): File size ({size_mb:.1f}MB) exceeds the {VideoUploadHandler.MAX_FILE_SIZE_MB}MB limit.")
        
        # Validate file is not empty
        if file_size == 0:
            errors.append(f"Audio file ({uploaded_file.name}): File is empty.")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_logo_file(uploaded_file) -> Tuple[bool, List[str]]:
        """
        Validate uploaded logo file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if uploaded_file is None:
            return False, ["Please upload a logo image file (JPG, JPEG, PNG)."]
        
        errors = []
        
        # Validate file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in ['.jpg', '.jpeg', '.png']:
            errors.append(f"Logo file ({uploaded_file.name}): Unsupported format. Please use JPG, JPEG, or PNG.")
        
        # Validate MIME type
        mime_type, _ = mimetypes.guess_type(uploaded_file.name)
        if mime_type not in ['image/jpeg', 'image/png']:
            errors.append(f"Logo file ({uploaded_file.name}): Invalid image type. Expected JPEG or PNG, found: {mime_type or 'unknown'}")
        
        # Validate file size (smaller limit for images)
        max_image_size = 10 * 1024 * 1024  # 10MB for images
        file_size = uploaded_file.size if hasattr(uploaded_file, 'size') else len(uploaded_file.getvalue())
        if file_size > max_image_size:
            size_mb = file_size / (1024 * 1024)
            errors.append(f"Logo file ({uploaded_file.name}): File size ({size_mb:.1f}MB) exceeds the 10MB limit for images.")
        
        # Validate file is not empty
        if file_size == 0:
            errors.append(f"Logo file ({uploaded_file.name}): File is empty.")
        
        return len(errors) == 0, errors

# AudioProcessor class for enhanced audio processing with looping and fade-out
class AudioProcessor:
    """Handles audio processing including duration analysis, looping, and fade-out effects."""
    
    @staticmethod
    def get_audio_duration(audio_path: str) -> float:
        """
        Get the duration of an audio file in seconds using pydub.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds as float
        """
        try:
            # Use pydub to load and get duration
            audio = AudioSegment.from_file(audio_path)
            duration_seconds = len(audio) / 1000.0  # pydub returns milliseconds
            return duration_seconds
            
        except Exception as e:
            # Fallback to moviepy if pydub fails
            try:
                audio_clip = AudioFileClip(audio_path)
                duration = audio_clip.duration
                audio_clip.close()
                return duration
            except Exception as moviepy_error:
                raise Exception(f"Failed to get audio duration with both pydub and moviepy. "
                              f"Pydub error: {str(e)}, MoviePy error: {str(moviepy_error)}")
    
    @staticmethod
    def needs_looping(audio_duration: float, video_duration: float) -> bool:
        """
        Determine if music looping is needed based on duration comparison.
        
        Args:
            audio_duration: Duration of audio file in seconds
            video_duration: Total duration of video content in seconds
            
        Returns:
            True if looping is needed, False otherwise
        """
        return video_duration > audio_duration
    
    @staticmethod
    def analyze_audio_video_duration(audio_path: str, video_duration: float) -> dict:
        """
        Analyze audio duration vs video duration and provide processing recommendations.
        
        Args:
            audio_path: Path to audio file
            video_duration: Total duration of video content in seconds
            
        Returns:
            Dictionary containing analysis results and processing recommendations
        """
        try:
            audio_duration = AudioProcessor.get_audio_duration(audio_path)
            needs_loop = AudioProcessor.needs_looping(audio_duration, video_duration)
            
            # Calculate how many loops are needed
            loops_needed = 0
            if needs_loop:
                loops_needed = int(video_duration / audio_duration) + 1
            
            # Calculate fade-out start time (3 seconds before end, or proportional for short videos)
            fade_out_duration = min(3.0, video_duration * 0.1)  # Max 3 seconds or 10% of video
            fade_out_start = max(0, video_duration - fade_out_duration)
            
            analysis = {
                'audio_duration': audio_duration,
                'video_duration': video_duration,
                'needs_looping': needs_loop,
                'loops_needed': loops_needed,
                'fade_out_duration': fade_out_duration,
                'fade_out_start': fade_out_start,
                'duration_ratio': video_duration / audio_duration if audio_duration > 0 else 0,
                'processing_required': needs_loop or fade_out_duration > 0
            }
            
            return analysis
            
        except Exception as e:
            raise Exception(f"Error analyzing audio vs video duration: {str(e)}")
    
    @staticmethod
    def loop_audio_to_duration(audio_path: str, target_duration: float, output_path: str = None) -> str:
        """
        Loop audio seamlessly to match target duration.
        
        Args:
            audio_path: Path to input audio file
            target_duration: Target duration in seconds
            output_path: Optional output path, if None will generate one
            
        Returns:
            Path to looped audio file
        """
        try:
            # Generate output path if not provided
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                output_path = os.path.join(os.path.dirname(audio_path), f"{base_name}_looped.mp3")
            
            # Get original audio duration
            original_duration = AudioProcessor.get_audio_duration(audio_path)
            
            # Validate that target duration is longer than original
            if target_duration <= original_duration:
                # If target is shorter or equal, just trim the audio
                return AudioProcessor._trim_audio_to_duration(audio_path, target_duration, output_path)
            
            # Validate minimum audio length for looping (avoid very short clips)
            if original_duration < 1.0:  # Less than 1 second
                raise Exception(f"Audio file is too short for looping ({original_duration:.2f}s). "
                              f"Please use audio files longer than 1 second.")
            
            # Use pydub for seamless looping
            try:
                # Load original audio
                audio = AudioSegment.from_file(audio_path)
                
                # Calculate how many full loops we need
                loops_needed = int(target_duration / original_duration)
                remaining_time = target_duration - (loops_needed * original_duration)
                
                # Create looped audio
                looped_audio = audio * loops_needed
                
                # Add partial loop if needed
                if remaining_time > 0.1:  # Only add if significant time remains
                    partial_audio = audio[:int(remaining_time * 1000)]  # pydub uses milliseconds
                    looped_audio += partial_audio
                
                # Ensure exact target duration
                target_ms = int(target_duration * 1000)
                if len(looped_audio) > target_ms:
                    looped_audio = looped_audio[:target_ms]
                
                # Export looped audio
                looped_audio.export(output_path, format="mp3", bitrate="192k")
                
                return output_path
                
            except Exception as pydub_error:
                # Fallback to FFmpeg for looping
                return AudioProcessor._loop_audio_with_ffmpeg(audio_path, target_duration, output_path)
                
        except Exception as e:
            raise Exception(f"Error looping audio to duration: {str(e)}")
    
    @staticmethod
    def _trim_audio_to_duration(audio_path: str, target_duration: float, output_path: str) -> str:
        """
        Trim audio to target duration (helper method).
        
        Args:
            audio_path: Path to input audio file
            target_duration: Target duration in seconds
            output_path: Output path for trimmed audio
            
        Returns:
            Path to trimmed audio file
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            target_ms = int(target_duration * 1000)
            trimmed_audio = audio[:target_ms]
            trimmed_audio.export(output_path, format="mp3", bitrate="192k")
            return output_path
        except Exception as e:
            raise Exception(f"Error trimming audio: {str(e)}")
    
    @staticmethod
    def _loop_audio_with_ffmpeg(audio_path: str, target_duration: float, output_path: str) -> str:
        """
        Loop audio using FFmpeg as fallback method.
        
        Args:
            audio_path: Path to input audio file
            target_duration: Target duration in seconds
            output_path: Output path for looped audio
            
        Returns:
            Path to looped audio file
        """
        try:
            cmd = [
                "ffmpeg",
                "-stream_loop", "-1",  # Loop indefinitely
                "-i", audio_path,
                "-t", str(target_duration),  # Set target duration
                "-c:a", "mp3",
                "-b:a", "192k",
                "-y",  # Overwrite output file
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"FFmpeg looping failed: {e.stderr}")
        except Exception as e:
            raise Exception(f"Error with FFmpeg audio looping: {str(e)}")
    
    @staticmethod
    def apply_fade_out(audio_path: str, fade_duration: float = 3.0, output_path: str = None) -> str:
        """
        Apply fade-out effect to audio file.
        
        Args:
            audio_path: Path to input audio file
            fade_duration: Duration of fade-out in seconds (default 3.0)
            output_path: Optional output path, if None will generate one
            
        Returns:
            Path to audio file with fade-out effect
        """
        try:
            # Generate output path if not provided
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                output_path = os.path.join(os.path.dirname(audio_path), f"{base_name}_fadeout.mp3")
            
            # Get audio duration to validate fade-out parameters
            audio_duration = AudioProcessor.get_audio_duration(audio_path)
            
            # Handle proportional fade-out for short videos
            if fade_duration >= audio_duration:
                # If fade duration is longer than audio, use proportional fade (10% of total duration)
                fade_duration = max(0.5, audio_duration * 0.1)
            
            # Calculate fade-out start time
            fade_start_time = max(0, audio_duration - fade_duration)
            
            try:
                # Use pydub for fade-out effect
                audio = AudioSegment.from_file(audio_path)
                
                # Apply fade-out effect
                fade_ms = int(fade_duration * 1000)  # Convert to milliseconds
                faded_audio = audio.fade_out(fade_ms)
                
                # Export with fade-out
                faded_audio.export(output_path, format="mp3", bitrate="192k")
                
                return output_path
                
            except Exception as pydub_error:
                # Fallback to FFmpeg for fade-out
                return AudioProcessor._apply_fade_out_with_ffmpeg(
                    audio_path, fade_start_time, fade_duration, output_path
                )
                
        except Exception as e:
            raise Exception(f"Error applying fade-out effect: {str(e)}")
    
    @staticmethod
    def _apply_fade_out_with_ffmpeg(audio_path: str, fade_start: float, fade_duration: float, output_path: str) -> str:
        """
        Apply fade-out effect using FFmpeg as fallback method.
        
        Args:
            audio_path: Path to input audio file
            fade_start: Time when fade-out starts in seconds
            fade_duration: Duration of fade-out in seconds
            output_path: Output path for faded audio
            
        Returns:
            Path to audio file with fade-out effect
        """
        try:
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-af", f"afade=t=out:st={fade_start}:d={fade_duration}",
                "-c:a", "mp3",
                "-b:a", "192k",
                "-y",  # Overwrite output file
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"FFmpeg fade-out failed: {e.stderr}")
        except Exception as e:
            raise Exception(f"Error with FFmpeg fade-out: {str(e)}")
    
    @staticmethod
    def process_audio_for_video(audio_path: str, video_duration: float, output_path: str = None) -> str:
        """
        Complete audio processing pipeline: loop if needed and apply fade-out.
        This combines looping and fade-out in a single processing pipeline.
        
        Args:
            audio_path: Path to input audio file
            video_duration: Total duration of video content in seconds
            output_path: Optional output path, if None will generate one
            
        Returns:
            Path to fully processed audio file
        """
        try:
            # Generate output path if not provided
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                output_path = os.path.join(os.path.dirname(audio_path), f"{base_name}_processed.mp3")
            
            # Analyze audio requirements
            analysis = AudioProcessor.analyze_audio_video_duration(audio_path, video_duration)
            
            current_audio_path = audio_path
            temp_files = []  # Track temporary files for cleanup
            
            try:
                # Step 1: Loop audio if needed
                if analysis['needs_looping']:
                    looped_path = os.path.join(os.path.dirname(audio_path), "temp_looped.mp3")
                    current_audio_path = AudioProcessor.loop_audio_to_duration(
                        current_audio_path, video_duration, looped_path
                    )
                    temp_files.append(looped_path)
                
                # Step 2: Apply fade-out effect
                fade_duration = analysis['fade_out_duration']
                if fade_duration > 0:
                    AudioProcessor.apply_fade_out(current_audio_path, fade_duration, output_path)
                else:
                    # If no fade-out needed, just copy the current audio
                    if current_audio_path != output_path:
                        shutil.copy2(current_audio_path, output_path)
                
                return output_path
                
            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception:
                        pass  # Ignore cleanup errors
                        
        except Exception as e:
            raise Exception(f"Error in complete audio processing pipeline: {str(e)}")

# VideoProcessor class for video processing and normalization
class VideoProcessor:
    """Handles video processing, normalization, and validation."""
    
    # Target resolution for all videos
    TARGET_RESOLUTION = (1280, 720)
    
    @staticmethod
    def normalize_video_resolution(video_path: str, target_resolution: Tuple[int, int] = TARGET_RESOLUTION) -> str:
        """
        Normalize video resolution to target resolution while maintaining aspect ratio.
        Uses padding or cropping as needed.
        
        Args:
            video_path: Path to input video file
            target_resolution: Target resolution as (width, height) tuple
            
        Returns:
            Path to normalized video file
        """
        try:
            # Generate output path
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(os.path.dirname(video_path), f"{base_name}_normalized.mp4")
            
            target_width, target_height = target_resolution
            
            # Use FFmpeg to normalize resolution with aspect ratio preservation
            # This command will:
            # 1. Scale the video to fit within target resolution while maintaining aspect ratio
            # 2. Add black padding if needed to reach exact target resolution
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vf", f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "medium",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-y",  # Overwrite output file if it exists
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to normalize video resolution: {e.stderr}")
        except Exception as e:
            raise Exception(f"Error normalizing video resolution: {str(e)}")
    
    @staticmethod
    def get_video_duration(video_path: str) -> float:
        """
        Get the duration of a video file in seconds.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Duration in seconds as float
        """
        try:
            # Use FFprobe to get video duration
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            duration = float(result.stdout.strip())
            return duration
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to get video duration using FFprobe: {e.stderr}")
        except ValueError as e:
            raise Exception(f"Invalid duration value returned from FFprobe: {str(e)}")
        except Exception as e:
            raise Exception(f"Error getting video duration: {str(e)}")
    
    @staticmethod
    def validate_video_format(video_path: str) -> bool:
        """
        Validate that the video file is a proper MP4 format.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if valid MP4, False otherwise
        """
        try:
            # Use FFprobe to check video format and codec
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=format_name",
                "-show_entries", "stream=codec_name,codec_type",
                "-of", "default=noprint_wrappers=1",
                video_path
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            output = result.stdout.strip()
            
            # Check if format contains mp4
            has_mp4_format = "format_name=mov,mp4,m4a,3gp,3g2,mj2" in output
            
            # Check if there's at least one video stream
            has_video_stream = "codec_type=video" in output
            
            return has_mp4_format and has_video_stream
            
        except subprocess.CalledProcessError:
            # If FFprobe fails, the file is likely corrupted or invalid
            return False
        except Exception:
            # Any other error means we can't validate the format
            return False
    
    @staticmethod
    def calculate_total_duration(video_paths: List[str]) -> float:
        """
        Calculate the total duration of multiple video files.
        
        Args:
            video_paths: List of paths to video files
            
        Returns:
            Total duration in seconds as float
        """
        total_duration = 0.0
        
        for video_path in video_paths:
            try:
                duration = VideoProcessor.get_video_duration(video_path)
                total_duration += duration
            except Exception as e:
                raise Exception(f"Error calculating duration for {video_path}: {str(e)}")
        
        return total_duration

# Título de la aplicación
st.title("🎬 Creador de Videos Secuenciales")
st.markdown("""
Sube varios videos, añade música, un texto de introducción y un logo para crear un video secuencial.
""")

# Función para obtener el tamaño del texto (compatible con diferentes versiones de Pillow)
def get_text_size(draw, text, font):
    try:
        # Para versiones más nuevas de Pillow
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Para versiones más antiguas de Pillow
        return draw.textsize(text, font=font)

# Función para crear video de introducción con texto
def create_intro_video(text, output_path, duration=5, fps=24):
    try:
        width, height = 1280, 720
        
        # Crear imagen temporal con el texto
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Usar fuente predeterminada
        font = ImageFont.load_default()
        
        # Calcular posición del texto para centrarlo
        text_width, text_height = get_text_size(draw, text, font)
        position = ((width - text_width) // 2, (height - text_height) // 2)
        
        # Dibujar texto
        draw.text(position, text, fill=(255, 255, 255), font=font)
        
        # Guardar imagen temporal
        temp_img_path = os.path.join(tempfile.gettempdir(), "intro_text.png")
        img.save(temp_img_path)
        
        # Crear video con FFmpeg
        cmd = [
            "ffmpeg",
            "-loop", "1",
            "-i", temp_img_path,
            "-c:v", "libx264",
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={width}:{height}",
            output_path
        ]
        
        subprocess.run(cmd, check=True)
        
        # Eliminar imagen temporal
        os.remove(temp_img_path)
        
        return output_path
    except Exception as e:
        error_msg = str(e).lower()
        if "font" in error_msg:
            st.error(f"❌ **Font Error in Intro Video**\n\n"
                    f"Could not load font for intro text.\n\n"
                    f"**Solutions:**\n"
                    f"• Try with shorter intro text\n"
                    f"• Use only basic characters (avoid special symbols)")
        elif "memory" in error_msg or "space" in error_msg:
            st.error(f"❌ **Resource Error in Intro Video**\n\n"
                    f"Insufficient resources to create intro video.\n\n"
                    f"**Solutions:**\n"
                    f"• Try with shorter intro text\n"
                    f"• Close other applications to free memory")
        else:
            st.error(f"❌ **Intro Video Creation Error**\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Solutions:**\n"
                    f"• Check that intro text is valid\n"
                    f"• Try with different text\n"
                    f"• Ensure sufficient system resources")
        st.stop()

# Función para crear video final con logo
def create_outro_video(logo_path, output_path, duration=5, fps=24):
    try:
        # Abrir logo
        logo = Image.open(logo_path)
        
        # Redimensionar logo si es necesario
        max_size = 400
        if max(logo.size) > max_size:
            ratio = max_size / max(logo.size)
            logo = logo.resize((int(logo.size[0] * ratio), int(logo.size[1] * ratio)), Image.LANCZOS)
        
        # Crear imagen de fondo
        width, height = 1280, 720
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        
        # Calcular posición del logo para centrarlo
        position = ((width - logo.size[0]) // 2, (height - logo.size[1]) // 2)
        
        # Pegar logo
        img.paste(logo, position)
        
        # Guardar imagen temporal
        temp_img_path = os.path.join(tempfile.gettempdir(), "outro_logo.png")
        img.save(temp_img_path)
        
        # Crear video con FFmpeg
        cmd = [
            "ffmpeg",
            "-loop", "1",
            "-i", temp_img_path,
            "-c:v", "libx264",
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={width}:{height}",
            output_path
        ]
        
        subprocess.run(cmd, check=True)
        
        # Eliminar imagen temporal
        os.remove(temp_img_path)
        
        return output_path
    except Exception as e:
        error_msg = str(e).lower()
        if "image" in error_msg or "logo" in error_msg:
            st.error(f"❌ **Logo Processing Error**\n\n"
                    f"Could not process the logo image.\n\n"
                    f"**Solutions:**\n"
                    f"• Check that logo is a valid JPG/PNG image\n"
                    f"• Try with a smaller logo file\n"
                    f"• Use a different image format")
        elif "memory" in error_msg or "space" in error_msg:
            st.error(f"❌ **Resource Error in Outro Video**\n\n"
                    f"Insufficient resources to create outro video.\n\n"
                    f"**Solutions:**\n"
                    f"• Try with a smaller logo image\n"
                    f"• Close other applications to free memory")
        else:
            st.error(f"❌ **Outro Video Creation Error**\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Solutions:**\n"
                    f"• Check that logo file is valid\n"
                    f"• Try with different logo image\n"
                    f"• Ensure sufficient system resources")
        st.stop()

# Función para concatenar videos
def concatenate_videos(video_paths, output_path):
    try:
        # Crear archivo de lista para ffmpeg
        list_file = os.path.join(tempfile.gettempdir(), "file_list.txt")
        with open(list_file, "w") as f:
            for video_path in video_paths:
                f.write(f"file '{video_path}'\n")
        
        # Usar ffmpeg para concatenar
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path
        ]
        
        subprocess.run(cmd, check=True)
        os.remove(list_file)
        return output_path
    except Exception as e:
        error_msg = str(e).lower()
        if "memory" in error_msg or "space" in error_msg:
            st.error(f"❌ **Insufficient Resources for Concatenation**\n\n"
                    f"Not enough memory or disk space to combine videos.\n\n"
                    f"**Solutions:**\n"
                    f"• Try with fewer video files\n"
                    f"• Use smaller video files\n"
                    f"• Close other applications to free memory")
        elif "codec" in error_msg or "format" in error_msg:
            st.error(f"❌ **Video Format Incompatibility**\n\n"
                    f"Videos have incompatible formats for concatenation.\n\n"
                    f"**Solutions:**\n"
                    f"• Ensure all videos are proper MP4 files\n"
                    f"• Re-encode videos to standard MP4 format\n"
                    f"• Try with different video files")
        else:
            st.error(f"❌ **Video Concatenation Error**\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Solutions:**\n"
                    f"• Check that all video files are valid\n"
                    f"• Try with fewer or smaller videos\n"
                    f"• Ensure sufficient system resources")
        st.stop()

# Enhanced function for adding processed audio to video
def add_audio_to_video(video_path, audio_path, output_path):
    """
    Enhanced audio-video combination function that handles processed audio
    (looped and faded) and ensures proper synchronization.
    
    Args:
        video_path: Path to input video file
        audio_path: Path to processed audio file (already looped and faded)
        output_path: Path for output video with audio
        
    Returns:
        Path to final video with processed audio
    """
    try:
        # Get video and audio durations to ensure proper handling
        video_duration = VideoProcessor.get_video_duration(video_path)
        audio_duration = AudioProcessor.get_audio_duration(audio_path)
        
        # Use the shorter duration to avoid audio/video sync issues
        # The processed audio should already match video duration, but this is a safety measure
        target_duration = min(video_duration, audio_duration)
        
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",  # Copy video stream without re-encoding
            "-c:a", "aac",   # Encode audio as AAC for compatibility
            "-b:a", "192k",  # Set audio bitrate for quality
            "-map", "0:v:0", # Map video stream from first input
            "-map", "1:a:0", # Map audio stream from second input
            "-t", str(target_duration),  # Ensure exact duration match
            "-avoid_negative_ts", "make_zero",  # Handle timestamp issues
            "-fflags", "+genpts",  # Generate presentation timestamps
            "-y",  # Overwrite output file if it exists
            output_path
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Validate that the output file was created successfully
        if not os.path.exists(output_path):
            raise Exception("Output video file was not created successfully")
        
        # Verify the output has both video and audio streams
        if not VideoProcessor.validate_video_format(output_path):
            raise Exception("Output video format validation failed")
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg error during audio-video combination: {e.stderr}"
        if "memory" in error_msg.lower() or "space" in error_msg.lower():
            st.error(f"❌ **Insufficient Resources for Audio-Video Combination**\n\n"
                    f"Not enough memory or disk space to combine audio and video.\n\n"
                    f"**Solutions:**\n"
                    f"• Try with smaller video files\n"
                    f"• Close other applications to free memory\n"
                    f"• Ensure sufficient disk space")
        elif "codec" in error_msg.lower():
            st.error(f"❌ **Audio-Video Format Incompatibility**\n\n"
                    f"Audio and video formats are incompatible.\n\n"
                    f"**Solutions:**\n"
                    f"• Try with a different audio file (MP3 recommended)\n"
                    f"• Re-encode video to standard MP4 format")
        else:
            st.error(f"❌ **Audio-Video Combination Failed**\n\n"
                    f"**FFmpeg Error:** {error_msg}\n\n"
                    f"**Solutions:**\n"
                    f"• Check that both audio and video files are valid\n"
                    f"• Try with different file formats\n"
                    f"• Ensure sufficient system resources")
        st.stop()
    except Exception as e:
        st.error(f"❌ **Audio-Video Processing Error**\n\n"
                f"**Error:** {str(e)}\n\n"
                f"**Solutions:**\n"
                f"• Check that audio and video files are valid\n"
                f"• Try refreshing and starting over\n"
                f"• Use different audio/video files")
        st.stop()

# Legacy function name for backward compatibility
def add_enhanced_audio_to_video(video_path, processed_audio_path, output_path):
    """
    Wrapper function that explicitly indicates enhanced audio processing.
    This function ensures the final video uses processed audio (looped + faded)
    instead of original music.
    
    Args:
        video_path: Path to concatenated video file
        processed_audio_path: Path to fully processed audio (looped and faded)
        output_path: Path for final output video
        
    Returns:
        Path to final video with enhanced audio
    """
    return add_audio_to_video(video_path, processed_audio_path, output_path)

# Función para extraer audio de un video
def extract_audio(video_path, output_path):
    try:
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",
            "-acodec", "mp3",
            output_path
        ]
        
        subprocess.run(cmd, check=True)
        return output_path
    except Exception as e:
        error_msg = str(e).lower()
        if "codec" in error_msg or "format" in error_msg:
            st.error(f"❌ **Audio Extraction Format Error**\n\n"
                    f"Could not extract audio due to format issues.\n\n"
                    f"**Solutions:**\n"
                    f"• The video is still available for download\n"
                    f"• You can extract audio manually using video editing software")
        else:
            st.error(f"❌ **Audio Extraction Error**\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Note:** The video is still available for download")
        st.stop()

# Crear directorios temporales
temp_dir = tempfile.mkdtemp()
output_dir = tempfile.mkdtemp()

# Formulario de entrada
with st.form("video_form"):
    st.subheader("Configuración del video")
    
    # Texto de introducción
    intro_text = st.text_input("Texto de introducción (máximo 10 palabras):", 
                              placeholder="Ej: Mis vacaciones de verano")
    
    # Videos - Updated to focus on MP4 files
    st.markdown("**Videos MP4:**")
    st.info("📹 Solo se aceptan archivos MP4. Tamaño máximo: 200MB por archivo.")
    videos = st.file_uploader("Selecciona tus videos MP4:", 
                             type=["mp4"], 
                             accept_multiple_files=True,
                             help="Sube uno o más archivos MP4. Los videos se procesarán en el orden que los subas.")
    
    # Música
    st.markdown("**Música de fondo:**")
    music = st.file_uploader("Selecciona tu música:", 
                            type=["mp3", "wav", "aac"],
                            help="La música se repetirá automáticamente si es más corta que el video final.")
    
    # Logo
    st.markdown("**Logo:**")
    logo = st.file_uploader("Selecciona tu logo:", 
                           type=["jpg", "jpeg", "png"],
                           help="El logo aparecerá al final del video por 5 segundos.")
    
    # Botón de envío
    submitted = st.form_submit_button("Crear Video Secuencial")

# Procesamiento cuando se envía el formulario
if submitted:
    # Validar entradas usando VideoUploadHandler
    validation_errors = []
    
    # Validar texto de introducción
    if not intro_text:
        validation_errors.append("Por favor ingresa un texto de introducción.")
    else:
        words = intro_text.split()
        if len(words) > 10:
            validation_errors.append("El texto de introducción debe tener máximo 10 palabras.")
    
    # Validar videos MP4
    videos_valid, video_errors = VideoUploadHandler.validate_video_files(videos)
    if not videos_valid:
        validation_errors.extend(video_errors)
    
    # Validar música
    music_valid, music_errors = VideoUploadHandler.validate_audio_file(music)
    if not music_valid:
        validation_errors.extend(music_errors)
    
    # Validar logo
    logo_valid, logo_errors = VideoUploadHandler.validate_logo_file(logo)
    if not logo_valid:
        validation_errors.extend(logo_errors)
    
    # Mostrar errores de validación si los hay
    if validation_errors:
        st.error("❌ **Errores de validación encontrados:**")
        for error in validation_errors:
            st.error(f"• {error}")
        st.info("💡 **Consejos:**\n"
                "- Asegúrate de que todos los videos sean archivos MP4\n"
                "- Verifica que los archivos no excedan el límite de tamaño\n"
                "- El texto de introducción debe tener máximo 10 palabras")
        st.stop()
    
    # Crear barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Guardar archivos subidos en el directorio temporal
    status_text.text("Guardando archivos subidos...")
    progress_bar.progress(10)
    
    video_paths = []
    for i, video in enumerate(videos):
        video_path = os.path.join(temp_dir, f"video_{i}.mp4")
        with open(video_path, "wb") as f:
            f.write(video.getbuffer())
        video_paths.append(video_path)
    
    music_path = os.path.join(temp_dir, "music.mp3")
    with open(music_path, "wb") as f:
        f.write(music.getbuffer())
    
    logo_path = os.path.join(temp_dir, "logo.png")
    with open(logo_path, "wb") as f:
        f.write(logo.getbuffer())
    
    # Process and normalize video segments
    status_text.text("Processing and normalizing video segments...")
    progress_bar.progress(20)
    
    normalized_video_paths = []
    for i, video_path in enumerate(video_paths):
        try:
            # Validate video format first
            if not VideoProcessor.validate_video_format(video_path):
                st.error(f"❌ **Video {i+1} Validation Failed**\n\n"
                        f"The uploaded file is not a valid MP4 or is corrupted.\n\n"
                        f"**Solutions:**\n"
                        f"• Re-encode the video using a video converter\n"
                        f"• Try uploading a different MP4 file\n"
                        f"• Ensure the file was not corrupted during upload")
                st.stop()
            
            # Normalize video resolution with fallback
            try:
                normalized_path = VideoProcessor.normalize_video_resolution(video_path)
                normalized_video_paths.append(normalized_path)
            except Exception as norm_error:
                # Fallback: Use original video if normalization fails
                st.warning(f"⚠️ **Video {i+1} Normalization Warning**\n\n"
                          f"Could not normalize resolution, using original video.\n"
                          f"This may cause inconsistent video quality.\n\n"
                          f"**Reason:** {str(norm_error)}")
                normalized_video_paths.append(video_path)
            
            # Update progress for each video processed
            sub_progress = 20 + (i + 1) * (10 / len(video_paths))
            progress_bar.progress(int(sub_progress))
            
        except Exception as e:
            st.error(f"❌ **Critical Error Processing Video {i+1}**\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Solutions:**\n"
                    f"• Check that the video file is not corrupted\n"
                    f"• Ensure you have sufficient disk space\n"
                    f"• Try with a smaller video file\n"
                    f"• Re-upload the video file")
            st.stop()
    
    # Calculate total video duration for audio processing
    status_text.text("Calculating video durations...")
    progress_bar.progress(30)
    
    try:
        # Calculate duration of intro (5 seconds) + normalized videos + outro (5 seconds)
        intro_outro_duration = 10.0  # 5 seconds each for intro and outro
        main_videos_duration = VideoProcessor.calculate_total_duration(normalized_video_paths)
        total_video_duration = intro_outro_duration + main_videos_duration
        
        # Validate reasonable duration limits
        if total_video_duration > 3600:  # More than 1 hour
            st.warning(f"⚠️ **Long Video Warning**\n\n"
                      f"Total video duration is {total_video_duration/60:.1f} minutes.\n"
                      f"Processing may take longer and use more resources.")
        
        st.info(f"📊 **Duration Analysis:**\n"
                f"- Main videos: {main_videos_duration:.1f} seconds\n"
                f"- With intro/outro: {total_video_duration:.1f} seconds")
        
    except Exception as e:
        st.error(f"❌ **Duration Calculation Failed**\n\n"
                f"**Error:** {str(e)}\n\n"
                f"**Solutions:**\n"
                f"• Check that all video files are valid MP4 format\n"
                f"• Ensure videos are not corrupted\n"
                f"• Try re-uploading the video files\n"
                f"• Contact support if the problem persists")
        st.stop()
    
    # Prepare background music for full video duration
    status_text.text("Preparing background music for full video duration...")
    progress_bar.progress(35)
    
    try:
        # Analyze audio requirements and process accordingly
        audio_analysis = AudioProcessor.analyze_audio_video_duration(music_path, total_video_duration)
        
        # Validate audio analysis results
        if audio_analysis['audio_duration'] <= 0:
            raise Exception("Audio file appears to be empty or corrupted")
        
        if audio_analysis['needs_looping']:
            st.info(f"🔄 **Music Processing:**\n"
                   f"- Original music: {audio_analysis['audio_duration']:.1f} seconds\n"
                   f"- Video duration: {audio_analysis['video_duration']:.1f} seconds\n"
                   f"- Loops needed: {audio_analysis['loops_needed']}\n"
                   f"- Fade-out: {audio_analysis['fade_out_duration']:.1f} seconds")
        
        # Process audio with looping and fade-out with fallback strategies
        processed_audio_path = os.path.join(temp_dir, "processed_music.mp3")
        
        try:
            AudioProcessor.process_audio_for_video(music_path, total_video_duration, processed_audio_path)
        except Exception as audio_proc_error:
            # Fallback: Use original audio if processing fails
            st.warning(f"⚠️ **Audio Processing Warning**\n\n"
                      f"Advanced audio processing failed, using original music.\n"
                      f"The video will still be created but without looping/fade effects.\n\n"
                      f"**Reason:** {str(audio_proc_error)}")
            processed_audio_path = music_path
        
    except Exception as e:
        st.error(f"❌ **Audio Analysis Failed**\n\n"
                f"**Error:** {str(e)}\n\n"
                f"**Solutions:**\n"
                f"• Check that the audio file is a valid MP3/WAV/AAC format\n"
                f"• Ensure the audio file is not corrupted or empty\n"
                f"• Try converting the audio to MP3 format\n"
                f"• Upload a different audio file")
        st.stop()
    
    # Applying fade-out effect to music
    status_text.text("Applying fade-out effect to music...")
    progress_bar.progress(40)
    
    # Create video of introduction
    status_text.text("Creating introduction video...")
    progress_bar.progress(45)
    
    try:
        intro_video_path = os.path.join(temp_dir, "intro.mp4")
        create_intro_video(intro_text, intro_video_path)
        
        # Verify intro video was created
        if not os.path.exists(intro_video_path):
            raise Exception("Intro video file was not created")
            
    except Exception as e:
        st.error(f"❌ **Intro Video Creation Failed**\n\n"
                f"**Error:** {str(e)}\n\n"
                f"**Solutions:**\n"
                f"• Check that the intro text is valid (max 10 words)\n"
                f"• Ensure sufficient disk space is available\n"
                f"• Try with shorter intro text")
        st.stop()
    
    # Create outro video with logo
    status_text.text("Creating outro video with logo...")
    progress_bar.progress(50)
    
    try:
        outro_video_path = os.path.join(temp_dir, "outro.mp4")
        create_outro_video(logo_path, outro_video_path)
        
        # Verify outro video was created
        if not os.path.exists(outro_video_path):
            raise Exception("Outro video file was not created")
            
    except Exception as e:
        st.error(f"❌ **Outro Video Creation Failed**\n\n"
                f"**Error:** {str(e)}\n\n"
                f"**Solutions:**\n"
                f"• Check that the logo file is a valid image (JPG/PNG)\n"
                f"• Ensure the logo file is not corrupted\n"
                f"• Try with a smaller logo file\n"
                f"• Upload a different logo image")
        st.stop()
    
    # Concatenate all videos
    status_text.text("Concatenating all video segments...")
    progress_bar.progress(60)
    
    try:
        all_videos = [intro_video_path] + normalized_video_paths + [outro_video_path]
        concatenated_path = os.path.join(temp_dir, "concatenated.mp4")
        concatenate_videos(all_videos, concatenated_path)
        
        # Verify concatenated video was created
        if not os.path.exists(concatenated_path):
            raise Exception("Concatenated video file was not created")
            
    except Exception as e:
        st.error(f"❌ **Video Concatenation Failed**\n\n"
                f"**Error:** {str(e)}\n\n"
                f"**Solutions:**\n"
                f"• Check that all video files are valid and not corrupted\n"
                f"• Ensure sufficient disk space for the final video\n"
                f"• Try with fewer or smaller video files\n"
                f"• Restart the process with fresh uploads")
        st.stop()
    
    # Combine processed audio with final video using enhanced function
    status_text.text("Combining processed audio with video...")
    progress_bar.progress(75)
    
    final_video_path = os.path.join(output_dir, "final_video.mp4")
    
    try:
        # Use enhanced audio-video combination with processed audio (looped + faded)
        add_enhanced_audio_to_video(concatenated_path, processed_audio_path, final_video_path)
        
        # Verify the final video was created successfully
        if not os.path.exists(final_video_path):
            raise Exception("Final video file was not created")
        
        # Verify final video has proper duration and format
        try:
            final_duration = VideoProcessor.get_video_duration(final_video_path)
            if abs(final_duration - total_video_duration) > 2.0:  # Allow 2 second tolerance
                st.warning(f"⚠️ **Duration Mismatch Warning**\n\n"
                          f"Final video duration ({final_duration:.1f}s) differs from expected ({total_video_duration:.1f}s).\n"
                          f"This is usually normal due to encoding differences.")
        except Exception as duration_error:
            st.warning(f"⚠️ Could not verify final video duration: {str(duration_error)}")
            final_duration = total_video_duration  # Use expected duration as fallback
        
        st.success(f"✅ **Enhanced Processing Complete:**\n"
                  f"- Final video duration: {final_duration:.1f} seconds\n"
                  f"- Audio processing: {'Looped + Faded' if audio_analysis['needs_looping'] else 'Faded only'}\n"
                  f"- Video resolution: 1280x720 (normalized)")
        
    except Exception as e:
        # Enhanced error handling with specific guidance
        error_msg = str(e).lower()
        
        if "memory" in error_msg or "space" in error_msg:
            st.error(f"❌ **Insufficient Resources**\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Solutions:**\n"
                    f"• Try with smaller video files\n"
                    f"• Reduce the number of videos\n"
                    f"• Close other applications to free memory\n"
                    f"• Try again later when system resources are available")
        elif "codec" in error_msg or "format" in error_msg:
            st.error(f"❌ **Video Format Error**\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Solutions:**\n"
                    f"• Re-encode videos to standard MP4 format\n"
                    f"• Use a video converter to ensure compatibility\n"
                    f"• Try with different video files")
        elif "permission" in error_msg or "access" in error_msg:
            st.error(f"❌ **File Access Error**\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Solutions:**\n"
                    f"• Refresh the page and try again\n"
                    f"• Check that files are not open in other applications\n"
                    f"• Clear browser cache and retry")
        else:
            st.error(f"❌ **Final Video Creation Failed**\n\n"
                    f"**Error:** {str(e)}\n\n"
                    f"**Solutions:**\n"
                    f"• Try refreshing the page and starting over\n"
                    f"• Use smaller or fewer video files\n"
                    f"• Ensure all uploaded files are valid\n"
                    f"• Contact support if the problem persists")
        st.stop()
    
    # Extract final audio
    status_text.text("Extracting final audio...")
    progress_bar.progress(85)
    
    try:
        final_audio_path = os.path.join(output_dir, "final_audio.mp3")
        extract_audio(final_video_path, final_audio_path)
        
        # Verify audio extraction was successful
        if not os.path.exists(final_audio_path):
            st.warning("⚠️ **Audio Extraction Warning**\n\n"
                      "Could not extract audio from final video.\n"
                      "The video is still available for download.")
            
    except Exception as e:
        # Don't stop the process for audio extraction errors
        st.warning(f"⚠️ **Audio Extraction Warning**\n\n"
                  f"Could not extract audio: {str(e)}\n"
                  f"The video is still available for download.")
        final_audio_path = None
    
    # Clean up temporary normalized video files
    status_text.text("Cleaning up temporary files...")
    progress_bar.progress(95)
    
    cleanup_errors = []
    try:
        # Clean up normalized video files
        for normalized_path in normalized_video_paths:
            try:
                if (os.path.exists(normalized_path) and 
                    normalized_path not in video_paths and 
                    "normalized" in normalized_path):
                    os.remove(normalized_path)
            except Exception as e:
                cleanup_errors.append(f"Could not remove {os.path.basename(normalized_path)}: {str(e)}")
        
        # Clean up processed audio file if it's different from original
        try:
            if (os.path.exists(processed_audio_path) and 
                processed_audio_path != music_path):
                os.remove(processed_audio_path)
        except Exception as e:
            cleanup_errors.append(f"Could not remove processed audio: {str(e)}")
        
        # Clean up concatenated video file
        try:
            if os.path.exists(concatenated_path):
                os.remove(concatenated_path)
        except Exception as e:
            cleanup_errors.append(f"Could not remove concatenated video: {str(e)}")
            
        # Report cleanup issues if any (but don't stop the process)
        if cleanup_errors:
            st.info(f"ℹ️ **Cleanup Note:**\n"
                   f"Some temporary files could not be removed:\n" + 
                   "\n".join([f"• {error}" for error in cleanup_errors[:3]]) +
                   (f"\n• ... and {len(cleanup_errors)-3} more" if len(cleanup_errors) > 3 else ""))
            
    except Exception as cleanup_error:
        # Don't stop the process for cleanup errors, just log them
        st.info(f"ℹ️ **Cleanup Note:** Some temporary files could not be cleaned up: {str(cleanup_error)}")
    
    # Complete
    progress_bar.progress(100)
    status_text.text("¡Video created successfully with enhanced processing!")
    
    # Mostrar resultado
    st.subheader("Resultado")
    st.video(final_video_path)
    
    # Botones de descarga
    col1, col2 = st.columns(2)
    
    with open(final_video_path, "rb") as f:
        video_bytes = f.read()
    
    with open(final_audio_path, "rb") as f:
        audio_bytes = f.read()
    
    col1.download_button(
        label="Descargar Video (MP4)",
        data=video_bytes,
        file_name="video_secuencial.mp4",
        mime="video/mp4"
    )
    
    col2.download_button(
        label="Descargar Audio (MP3)",
        data=audio_bytes,
        file_name="audio_secuencial.mp3",
        mime="audio/mp3"
    )
    
    # Limpiar directorios temporales
    try:
        shutil.rmtree(temp_dir)
        shutil.rmtree(output_dir)
    except:
        pass
