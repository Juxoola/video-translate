import gradio as gr
import os
import subprocess
import uuid
from TTS.api import TTS
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import logging
import shutil
import torch
import time
from pydub import AudioSegment
from pydub.effects import speedup
import re

# Конфигурация логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Конфигурация приложения
UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/output'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def is_cuda_available():
    # Проверяет наличие CUDA.
    return torch.cuda.is_available()

# Конфигурация Faster-Whisper
#MODEL_SIZE = "tiny"
MODEL_SIZE = "large-v3"

if is_cuda_available():
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="int8_float16")
else:
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

# Получение устройства
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def adjust_audio_tempo_ffmpeg(input_path, output_path, tempo_factor):
    # Меняет темп аудио с сохранением высоты тона через FFmpeg.
    # Если коэффициент вне диапазона 0.5–2.0, он делится на нужные части.
    factors = []
    temp_factor = tempo_factor
    # Разбивка на коэффициенты не более 2.0
    while temp_factor > 2.0:
        factors.append(2.0)
        temp_factor /= 2.0
    while temp_factor < 0.5:
        factors.append(0.5)
        temp_factor /= 0.5
    factors.append(temp_factor)
    
    atempo_filters = ",".join("atempo={:.3f}".format(f) for f in factors)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter:a", atempo_filters,
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def parse_vtt_time(time_str):
    # Преобразует строку времени VTT в секунды.
    hours, minutes, seconds = time_str.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

def generate_audio_segment(text, start_time, end_time, speaker_wav, language, output_dir):
    # Генерирует аудио для переведённого текста.
    temp_audio_path = os.path.join(output_dir, f"temp_{uuid.uuid4()}.wav")
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=temp_audio_path)

    audio = AudioSegment.from_wav(temp_audio_path)
    target_duration = (end_time - start_time) * 1000

    if audio.duration_seconds * 1000 > target_duration:
        # Вычисляем требуемый коэффициент ускорения
        playback_speed = audio.duration_seconds * 1000 / target_duration
        playback_speed = min(playback_speed, 1.8)
        adjusted_audio_path = temp_audio_path.replace(".wav", "_adjusted.wav")
        adjust_audio_tempo_ffmpeg(temp_audio_path, adjusted_audio_path, playback_speed)
        audio = AudioSegment.from_wav(adjusted_audio_path)
        os.remove(adjusted_audio_path)
        
        # Если даже после изменения темпа аудио всё еще длиннее,
        # обрезаем его до требуемой длительности.
        if audio.duration_seconds * 1000 > target_duration:
            audio = audio[:target_duration]
    elif audio.duration_seconds * 1000 < target_duration:
        # Добавляем тишину, если аудио короче требуемой длительности
        silence_duration = target_duration - audio.duration_seconds * 1000
        silence = AudioSegment.silent(duration=silence_duration)
        audio = audio + silence

    audio.export(temp_audio_path, format="wav")
    return temp_audio_path, audio

def translate_vtt(vtt_content, target_language='en', output_dir=OUTPUT_FOLDER, vtt_filepath=None, original_audio_path=None):
    # Переводит VTT-файл, генерирует звуковые сегменты и объединяет их в итоговое аудио.
    if not vtt_content or not isinstance(vtt_content, str):
        return vtt_content, None

    lines = vtt_content.splitlines()
    translated_lines = []
    audio_segments = []
    
    if not lines[0].startswith('WEBVTT'):
        translated_lines.append('WEBVTT')
        translated_lines.append('') 
    else:
        translated_lines.append(lines[0])
        translated_lines.append('')
        

    # Находим время начала первого субтитра
    first_subtitle_start_time = None
    for line in lines:
        if '-->' in line:
            start_time_str = line.split('-->')[0].strip()
            first_subtitle_start_time = parse_vtt_time(start_time_str)
            break

    speaker_wav = extract_speaker_sample(original_audio_path, output_dir) if original_audio_path else None

    for i, line in enumerate(lines):
        if i == 0:
            continue
        if line.startswith('NOTE'):
            translated_lines.append(line)
            continue
        if '-->' in line:
            start_time_str, end_time_str = line.split('-->')
            start_time = parse_vtt_time(start_time_str.strip())
            end_time = parse_vtt_time(end_time_str.strip())

            text_line_index = i + 1
            if text_line_index < len(lines):
                text_to_translate = lines[text_line_index].strip()
                try:
                    translated_text = GoogleTranslator(source='auto', target=target_language).translate(text_to_translate)
                except Exception as e:
                    logging.error(f"Ошибка перевода: {e}")
                    translated_text = text_to_translate
                translated_lines.append(line)
                translated_lines.append(translated_text)
                translated_lines.append("")
                if speaker_wav:
                    audio_segment_path, audio_segment = generate_audio_segment(
                        translated_text, start_time, end_time, speaker_wav, target_language, output_dir
                    )
                    audio_segments.append((start_time, audio_segment_path, audio_segment))

    translated_vtt_content = '\n'.join(translated_lines)

    # Объединяем аудиосегменты с соответствующими паузами
    final_audio = AudioSegment.empty()
    last_end_time = 0

    # Добавляем начальную тишину, если первый субтитр не начинается с 0
    if first_subtitle_start_time > 0:
        initial_silence = AudioSegment.silent(duration=first_subtitle_start_time * 1000)
        final_audio += initial_silence
        last_end_time = first_subtitle_start_time

    for start_time, audio_path, audio_segment in audio_segments:
        silence_duration = (start_time - last_end_time) * 1000
        if silence_duration > 0:
            silence = AudioSegment.silent(duration=silence_duration)
            final_audio += silence

        final_audio += audio_segment
        last_end_time = start_time + audio_segment.duration_seconds

    # Сохраняем итоговый переведенный аудиофайл
    if vtt_filepath:
        translated_audio_filename = f"{os.path.splitext(os.path.basename(vtt_filepath))[0]}_{target_language}.wav"
    else:
        translated_audio_filename = f"translated_{target_language}.wav"
    translated_audio_filepath = os.path.join(output_dir, translated_audio_filename)
    final_audio.export(translated_audio_filepath, format="wav")
    
    for _, audio_path, _ in audio_segments:
        try:
            os.remove(audio_path)
        except Exception as e:
            logging.error(f"Ошибка удаления промежуточного аудиофайла {audio_path}: {e}")

    logging.info(f"Переведенная аудиодорожка сгенерирована: {translated_audio_filepath}")
    return translated_vtt_content, translated_audio_filepath

def extract_speaker_sample(video_path, output_dir, duration=10):
    # Извлекает аудио-образец из видео.
    try:
        sample_path = os.path.join(output_dir, "speaker_sample.wav")
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", "0",
            "-t", str(duration),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "22050",
            "-ac", "1",
            sample_path
        ]
        if is_cuda_available():
            cmd.insert(1, "-hwaccel")
            cmd.insert(2, "cuda")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"Образец аудио для диктора извлечен: {sample_path}")
        return sample_path
    except Exception as e:
        logging.error(f"Ошибка извлечения образца аудио: {e}")
        return None

def extract_text_for_tts(vtt_content):
    lines = vtt_content.splitlines()
    text_lines = []
    for line in lines:
        if not '-->' in line and not line.startswith('WEBVTT') and not line.startswith('NOTE') and line.strip():
            text_lines.append(line)
    return '\n'.join(text_lines)

def transcribe(video_file, output_dir):
    # Транскрибирует видео через Whisper и сохраняет результаты.
    start_time = time.time()
    full_media_path = None  
    full_audio_path = None 
    if not video_file:
        return "Ошибка: Не предоставлен видеофайл.", None, None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "Ошибка: Не предоставлен видеофайл."

    if not output_dir:
        output_dir = OUTPUT_FOLDER

    try:
        file_type = "video"

        unique_filename = str(uuid.uuid4())
        filename = unique_filename + os.path.splitext(video_file)[1]
        full_media_path = os.path.join(UPLOAD_FOLDER, filename)

        # Копируем файл вместо переименования
        shutil.copy2(video_file, full_media_path)
        logging.info(f"Видеофайл скопирован во временное хранилище: {full_media_path}")

        original_filename = os.path.splitext(os.path.basename(video_file))[0]
        wav_filename = original_filename + ".wav"
        full_wav_path = os.path.join(output_dir, wav_filename)

        ffmpeg_command = [
            "ffmpeg", "-i", full_media_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", full_wav_path
        ]
        if is_cuda_available():
            ffmpeg_command.insert(1, "-hwaccel")
            ffmpeg_command.insert(2, "cuda")
        subprocess.run(ffmpeg_command, check=True)
        logging.info(f"Файл конвертирован в WAV: {full_wav_path}")

        audio_filename = original_filename + "_audio.wav"
        full_audio_path = os.path.join(output_dir, audio_filename)
        ffmpeg_audio_command = [
            "ffmpeg",
            "-i", full_media_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "22050",
            "-ac", "1",
            full_audio_path
        ]
        if is_cuda_available():
            ffmpeg_audio_command.insert(1, "-hwaccel")
            ffmpeg_audio_command.insert(2, "cuda")
        subprocess.run(ffmpeg_audio_command, check=True)
        logging.info(f"Аудиодорожка извлечена: {full_audio_path}")

        # Транскрипция с помощью Faster-Whisper
        segments, info = model.transcribe(full_wav_path, beam_size=7, language="ru", vad_filter=True)
        logging.info("Определен язык '%s' с вероятностью %f", info.language, info.language_probability)

        vtt_content = ""
        transcript = ""
        for segment in segments:
            start_time_seg = "{:02d}:{:02d}:{:06.3f}".format(
                int(segment.start // 3600),
                int((segment.start % 3600) // 60),
                segment.start % 60
            )
            end_time_seg = "{:02d}:{:02d}:{:06.3f}".format(
                int(segment.end // 3600),
                int((segment.end % 3600) // 60),
                segment.end % 60
            )
            vtt_content += f"{start_time_seg} --> {end_time_seg}\n{segment.text}\n\n"
            transcript += f"{segment.text} "
            logging.info("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

        # Сохранение VTT и TXT файлов
        vtt_filepath = os.path.join(output_dir, original_filename + ".vtt")
        with open(vtt_filepath, "w", encoding="utf-8") as vtt_file:
            vtt_file.write("WEBVTT\n\n")
            vtt_file.write(vtt_content)
        transcript_filepath = os.path.join(output_dir, original_filename + ".txt")
        with open(transcript_filepath, "w", encoding="utf-8") as txt_file:
            txt_file.write(transcript)

        logging.info(f"Copied file remains: {full_media_path}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logging.info("Транскрипция успешно завершена")
        transcribe_status_message = f"Транскрипция успешно завершена. Затрачено времени: {elapsed_time:.2f} секунд"

        return (
            transcript,
            vtt_content,
            vtt_filepath,
            transcript_filepath,
            vtt_filepath,
            full_media_path,
            gr.update(visible=True),
            gr.update(visible=True), 
            gr.update(visible=True), 
            gr.update(visible=True), 
            gr.update(visible=True), 
            gr.update(visible=True),
            gr.update(visible=False), 
            full_audio_path,
            transcribe_status_message,
            gr.update(visible=True),
        )

    except Exception as e:
        logging.error(f"Ошибка во время транскрипции: {e}")
        return f"Ошибка: {e}", None, None, None, None, full_media_path, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), None, f"Ошибка во время транскрипции: {e}",gr.update(visible=False)

def save_vtt(vtt_content, vtt_filepath):
    try:
        with open(vtt_filepath, "w", encoding="utf-8") as f:
            f.write(vtt_content)
        logging.info("VTT файл успешно сохранен")
        return "VTT файл успешно сохранен."
    except Exception as e:
        logging.error(f"Ошибка сохранения VTT файла: {e}")
        return f"Ошибка: {e}"

def get_audio_duration(filepath):
    # Возвращает длительность аудио в секундах.
    try:
        cmd = [
            "ffprobe",
            "-i", filepath,
            "-show_entries", "format=duration",
            "-v", "quiet",
            "-of", "csv=p=0"
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logging.error(f"Ошибка получения длительности аудио {filepath}: {e}")
        return 0.0

def extract_last_frame(video_path, image_path):
    # Извлекает последний кадр видео.
    try:
        cmd = [
            "ffmpeg",
            "-sseof", "-0.1",
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            image_path
        ]
        if is_cuda_available():
            cmd.insert(1, "-hwaccel")
            cmd.insert(2, "cuda")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        logging.error(f"Ошибка извлечения последнего кадра из {video_path}: {e}")

def create_freeze_frame_video(image_path, duration, video_path):
    # Создает видео из одного изображения.
    try:
        cmd = [
            "ffmpeg",
            "-loop", "1",
            "-i", image_path,
            "-c:v", "libx264",
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            "-vf", "scale=1280:720",
            video_path
        ]
        if is_cuda_available():
            cmd.insert(1, "-hwaccel")
            cmd.insert(2, "cuda")
            cmd[cmd.index("-c:v") + 1] = "h264_nvenc" 
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        logging.error(f"Ошибка создания стоп-кадра из {image_path}: {e}")

def concatenate_videos(video1, video2, output_path):
    # Объединяет два видеофайла.
    try:
        concat_file = os.path.join(os.path.dirname(output_path), "concat_list.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            f.write(f"file '{os.path.abspath(video1)}'\n")
            f.write(f"file '{os.path.abspath(video2)}'\n")
        
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            output_path
        ]
        if is_cuda_available():
            cmd.insert(1, "-hwaccel")
            cmd.insert(2, "cuda")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.remove(concat_file)
    except Exception as e:
        logging.error(f"Ошибка объединения видео {video1} и {video2}: {e}")

def translate_and_show(vtt_filepath, output_dir, translation_language, original_video, full_audio_path):
    # Переводит VTT, генерирует аудио и вставляет субтитры в видео.
    translate_start_time = time.time()
    if not vtt_filepath:
        return "Ошибка: Нет VTT файла для перевода.", None, None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "Ошибка: Нет VTT файла для перевода."

    if not output_dir:
        output_dir = OUTPUT_FOLDER

    if not original_video:
        return "Ошибка: Не предоставлен видеофайл.", None, None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "Ошибка: Не предоставлен видеофайл."
    
    try:
        with open(vtt_filepath, "r", encoding="utf-8") as f:
            vtt_content = f.read()

        translated_vtt_content, translated_audio_filepath = translate_vtt(vtt_content, translation_language, output_dir, vtt_filepath, full_audio_path)
        logging.info("VTT файл успешно переведен")

        original_filename = os.path.splitext(os.path.basename(vtt_filepath))[0]
        translated_vtt_filepath = os.path.join(output_dir, f"{original_filename}_{translation_language}.vtt")
        with open(translated_vtt_filepath, "w", encoding="utf-8") as f:
            f.write(translated_vtt_content)
        logging.info(f"Переведенный VTT сохранен: {translated_vtt_filepath}")

        interim_video_path = os.path.join(output_dir, f"{original_filename}_{translation_language}_interim.mp4")
        if translated_audio_filepath:
            ffmpeg_command = [
                "ffmpeg",
                "-i", original_video,
                "-i", translated_audio_filepath,
                "-c:v", "libx264",
                "-vf", f"subtitles={translated_vtt_filepath}",
                "-c:a", "aac",
                "-map", "0:v",
                "-map", "1:a",
                "-movflags", "+faststart",
                "-y",
                interim_video_path
            ]
            if is_cuda_available():
                ffmpeg_command.insert(1, "-hwaccel")
                ffmpeg_command.insert(2, "cuda")
                ffmpeg_command[ffmpeg_command.index("-c:v") + 1] = "h264_nvenc" 
            subprocess.run(ffmpeg_command, check=True)
            logging.info(f"Промежуточное видео с переведенной аудиодорожкой и субтитрами сохранено: {interim_video_path}")

            translated_audio_duration = get_audio_duration(translated_audio_filepath)
            video_duration = get_audio_duration(original_video)

            if translated_audio_duration > video_duration:
                extra_duration = translated_audio_duration - video_duration
                logging.info(f"Переведённое аудио длиннее на {extra_duration:.2f} секунд. Добавляется стоп-кадр.")

                last_frame_path = os.path.join(output_dir, "last_frame.png")
                freeze_video_path = os.path.join(output_dir, "freeze_frame.mp4")
                final_output_path = os.path.join(output_dir, f"{original_filename}_{translation_language}_output.mp4")

                extract_last_frame(original_video, last_frame_path)
                create_freeze_frame_video(last_frame_path, extra_duration, freeze_video_path)
                concatenate_videos(interim_video_path, freeze_video_path, final_output_path)
                output_video_path = final_output_path

                os.remove(last_frame_path)
                os.remove(freeze_video_path)
                os.remove(interim_video_path)

                logging.info(f"Итоговое видео с добавленным стоп-кадром сохранено: {output_video_path}")
            else:
                output_video_path = interim_video_path

            translate_end_time = time.time()
            translate_elapsed_time = translate_end_time - translate_start_time

            logging.info("Перевод успешно завершен")
            translate_status_message = f"Перевод успешно завершен. Затрачено времени: {translate_elapsed_time:.2f} секунд"

            return (
                translated_vtt_content,
                translated_vtt_filepath,
                translated_audio_filepath,
                output_video_path,
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True) if translated_audio_filepath else gr.update(visible=False),
                gr.update(visible=True),
                translate_status_message
            )
    except Exception as e:
        logging.error(f"Ошибка во время перевода: {e}")
        return f"Ошибка: {e}", None, None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), f"Ошибка во время перевода: {e}"

def cleanup_temp_files(original_video, full_media_path):
    # Удаляет временные файлы.
    try:
        if original_video and os.path.exists(original_video):
            os.remove(original_video)
            logging.info(f"Удален временный файл: {original_video}")
        if full_media_path and os.path.exists(full_media_path):
            os.remove(full_media_path)
            logging.info(f"Удален временный файл: {full_media_path}")
    except Exception as e:
        logging.error(f"Ошибка при удалении временных файлов: {e}")

with gr.Blocks(title="Транскрипция аудио с помощью Whisper") as demo:
    gr.Markdown("# Транскрипция видео с помощью Whisper")
    with gr.Row():
        video_input = gr.Video(label="Загрузить видео")
        output_dir_input = gr.Textbox(label="Папка для сохранения", value=OUTPUT_FOLDER)
    transcribe_button = gr.Button("Транскрибировать")
    transcribe_status_output = gr.Textbox(label="Статус транскрипции", interactive=False)
    with gr.Row(visible=False) as transcript_row:
        transcript_output = gr.Textbox(label="Транскрипция", lines=5, max_lines=100)
    with gr.Row(visible=False) as vtt_row:
        vtt_output = gr.Textbox(label="Оригинальные VTT субтитры", lines=5, max_lines=100)
        vtt_filepath_store = gr.Textbox(label="Путь к VTT файлу", visible=False)
    with gr.Row(visible=False) as vtt_edit_row:
        save_vtt_button = gr.Button("Сохранить VTT")
        save_vtt_message = gr.Textbox(label="Статус сохранения", interactive=False)
    with gr.Row(visible=False) as download_files_row:
        download_txt = gr.File(label="Скачать TXT")
        download_vtt = gr.File(label="Скачать VTT")
    with gr.Row(visible=False) as translate_button_row:
        translate_button = gr.Button("Перевести")
    with gr.Row(visible=False) as translate_lang_row:
        translate_lang = gr.Dropdown(
            label="Язык перевода",
            choices=["en", "de", "fr", "es", "it", "ja", "zh-CN", "ru"],
            value="en",
        )
    translate_status_output = gr.Textbox(label="Статус перевода", interactive=False, visible=False)
    with gr.Row(visible=False) as translated_vtt_row:
        translated_vtt_output = gr.Textbox(label="Переведенные VTT субтитры", lines=5, max_lines=100)
    with gr.Row(visible=False) as download_translated_files_row:
        download_translated_vtt = gr.File(label="Скачать переведенный VTT")
        download_translated_audio = gr.Audio(label="Прослушать перевод")
    with gr.Row(visible=False) as output_video_row:
        output_video = gr.Video(label="Видео с переводом")

    full_media_path_store = gr.Textbox(visible=False)
    full_audio_path_store = gr.Textbox(visible=False)

    transcribe_button.click(
        transcribe,
        inputs=[video_input, output_dir_input],
        outputs=[
            transcript_output,
            vtt_output,
            vtt_filepath_store,
            download_txt,
            download_vtt,
            full_media_path_store,
            transcript_row,
            vtt_row,
            vtt_edit_row,
            download_files_row,
            translate_button_row,
            translate_lang_row,
            translated_vtt_row,
            full_audio_path_store,
            transcribe_status_output,
            translate_status_output,
        ],
    )

    save_vtt_button.click(
        save_vtt,
        inputs=[vtt_output, vtt_filepath_store],
        outputs=[save_vtt_message]
    )

    translate_button.click(
        translate_and_show,
        inputs=[vtt_filepath_store, output_dir_input, translate_lang, full_media_path_store, full_audio_path_store],
        outputs=[
            translated_vtt_output,
            download_translated_vtt,
            download_translated_audio,
            output_video,
            download_translated_files_row,
            translated_vtt_row,
            download_translated_files_row,
            output_video_row,
            translate_status_output
        ],
    ).then(
        cleanup_temp_files,
        inputs=[video_input, full_media_path_store]
    )

    gr.Markdown("© 2024 Долгожданный перевод порнороликов, чтобы наконец-то понять их сюжеты")

demo.launch()