import initial
import os
import time
import shutil
from pydub import AudioSegment
from pydub.silence import split_on_silence
import torch
from moviepy.editor import VideoFileClip, AudioFileClip
import soundfile as sf
import subprocess
import io
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

Video_Flag = 0
n = 0

# 실행될 부분
def main_part(word_list):
    global Video_Flag
    global n
    global video_file
    global audio_file

    while True:

        if not os.path.isfile(initial.video_path + f"/video_{n}.mp4") and Video_Flag <= 40:  # 파일이 존재하지 않는 경우
            time.sleep(1)
            Video_Flag += 1
            continue
        elif os.path.isfile(initial.video_path + f"/video_{n}.mp4") and Video_Flag <= 40:  # 파일이 존재하는 경우
            video_file = initial.video_path + f"/video_{n}.mp4"
            video_to_audio(video_file)  # 오디오 파일 생성
            audio_file = initial.audio_path + f"/audio_{n}.wav"
            Video_Flag = 0
            video_conversion(word_list)
            clear_temp_directory() # temp 폴더 비우기, 중간 파일이 궁금하면 주석처리
            n += 1
        elif Video_Flag > 40:
            break

# Audio file 마다 수행
def video_conversion(word_list):
    transcription = audio_to_text(audio_file)  # transcription 생성
    target_dict_sorted = sorted(word_list, key=len, reverse=True) # 큰 단어를 먼저 수행하기 위함
    sent_dict = find_target(transcription, target_dict_sorted)  # 타겟 단어 있는지 확인 후 분리 문장 생성

    # 타임스탬프 추출
    timestamp_list = []
    for word in target_dict_sorted:
        timestamps = extract_timestamps(transcription, word, timestamp_list, target_dict_sorted)
        if timestamps:
            timestamp_list.append(timestamps)

    # 후처리를 위한 평균 데시벨 추출
    global Before_DB
    Before_DB = AudioSegment.from_file(audio_file).dBFS
    cnt = 0
    print("[sorted dict] :", target_dict_sorted)
    print("[timestampe_list] :", timestamp_list)

    alter_list = []
    replace_timestamp = []
    # 욕설마다 Replace 처리
    for timestamps in timestamp_list:
        for timestamp in timestamps:
            extract_target_audio(timestamp, cnt)
            candidates = predict_next_word(sent_dict[cnt][1], sent_dict[cnt][2], target_dict_sorted)
            best_cand = select_best_candidate(candidates, sent_dict[cnt][1], sent_dict[cnt][2])

            alter_list.append(best_cand)
            replace_timestamp.append(timestamp)

            # 음성 학습 및 대체음성 생성, 원본과 유사하도록 조정
            print("[대체어 후보 :", candidates,"]")
            print("[앞 문장 :", sent_dict[cnt][1], "]")
            print("[뒷 문장 :", sent_dict[cnt][2], "]")
            print("[대체해야 할 단어 :", sent_dict[cnt][0], "]")
            print("[선정한 대체어 :", best_cand, "]" )
            cnt += 1

    print("대체어 모음 :", alter_list, "" )
    if len(alter_list) > 0:
        voice_return(alter_list, initial.tts_model, initial.tone_color_converter)

        for i in range(len(alter_list)):
            voice_processing(initial.temp_path + f"/extracted_segment_{i}.wav", initial.temp_path + f"/audio_token_{i}.wav", initial.temp_path + f"/replace_audio_{i}.wav")
            # voice_processing은 아직 개선이 필요. voice_processing : 원본 음성과 데시벨, 속도 맞추기
            # 사용하길 원치 않는다면 주석처리 후 아래 코드 주석해제
            #AudioSegment.from_file(initial.temp_path + f"/audio_token_{i}.wav").export(initial.temp_path + f"/replace_audio_{i}.wav", format="wav")
            if not os.path.exists(initial.audio_path + f"/final_audio_{n}.wav"): # 이미 만들어진 최종 wav파일 없을 떄
                replace_audio_segment(audio_file, initial.temp_path + f"/replace_audio_{i}.wav", replace_timestamp[i], initial.audio_path + f"/final_audio_{n}.wav")
            else:
                replace_audio_segment(initial.audio_path + f"/final_audio_{n}.wav", initial.temp_path + f"/replace_audio_{i}.wav", replace_timestamp[i], initial.audio_path + f"/final_audio_{n}.wav")


    # 최종 음성을 영상에 합성
    if not os.path.exists(initial.audio_path + f"/final_audio_{n}.wav"):
        shutil.copy(video_file, initial.output_path + f"/final_video_{n}.mp4")
    else:
        replace_video(video_file, initial.audio_path + f"/final_audio_{n}.wav", initial.output_path + f"/final_video_{n}.mp4")

# Audio 추출
def video_to_audio(video_path):
    audio_path = initial.audio_path + f"/audio_{n}.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-q:a", "0",
        "-map", "a",
        audio_path
    ]
    subprocess.run(command, check=True)

# transcription 생성
def audio_to_text(audio_path):
    result = initial.wisper_model.transcribe(audio_path, verbose=True, language='ko')
    return result

# 타겟 단어 있는지 확인 후 분리 문장 생성
def find_target(result, target_dict):
    sentence = "".join(segment["text"] for segment in result["segments"])
    sent_dict = []
    found_indices = set()

    for word in target_dict:
        index = sentence.find(word)
        while index != -1:
            # 이전에 발견된 단어와 중복되지 않는지 확인
            if not any(start <= index < end for start, end in found_indices):
                before = sentence[:index]
                after = sentence[index + len(word):]
                sent_dict.append([word, before, after])
                found_indices.add((index, index + len(word)))
            index = sentence.find(word, index + 1)
    
    return sent_dict

# 타임스탬프 추출
def extract_timestamps(transcription, target_word, existing_timestamps, target_dict_sorted):
    timestamps = []
    for segment in transcription['segments']:
        text = segment['text']
        start_time = segment['start']
        end_time = segment['end']
        duration = (end_time - start_time) / len(text)
        index = text.find(target_word)
        while index != -1:
            word_start_time = start_time + index * duration - 0.1 # 파라미터 조정
            word_end_time = word_start_time + len(target_word) * duration -0.1 # 파라미터 조정
            adjustment = 0.1  # 추가 조정할 시간 (초)
            word_start_time = max(word_start_time - adjustment, start_time)
            word_end_time = min(word_end_time + adjustment, end_time)

            # 기존 타임스탬프와 겹치는지 확인
            # '씨발 새끼'를 이미 찾은 경우 '씨발'과 '새끼'를 추가적으로 찾지 않기 위함
            overlaps = False
            mask = False
            for longer_word in target_dict_sorted[:target_dict_sorted.index(target_word)]:
                if target_word in longer_word:
                    mask = True
                    break
            
            if mask:
                for ind in existing_timestamps:
                    for existing_start, existing_end in ind:
                        if not (word_end_time < existing_start or word_start_time > existing_end):
                            overlaps = True
                            break
            
            if not overlaps:
                timestamps.append((word_start_time, word_end_time))

            index = text.find(target_word, index + 1)
    return timestamps


# 타임스탬프 사이의 음성을 추출하여 새로운 파일로 저장
def extract_audio_segment(start_time, end_time, output_audio):
    audio = AudioSegment.from_wav(audio_file)
    start_ms = start_time * 1000  # 밀리초 단위
    end_ms = end_time * 1000  # 밀리초 단위
    audio_segment = audio[start_ms:end_ms]
    audio_segment.export(output_audio, format="wav")

# 타겟 오디오 추출
def extract_target_audio(timestamp, cnt):
    output_audio = initial.temp_path + f"/extracted_segment_{cnt}.wav"
    extract_audio_segment(timestamp[0], timestamp[1], output_audio)

# 한글 자음이나 모음만 있는 경우를 제외하기 위함
def is_hangul_syllable(word):
    for char in word:
        if not ('가' <= char <= '힣'):
            return False
    return True

# 대체어 예측 함수
def predict_next_word(first_part, second_part, word_list):
    text = f"{first_part}[MASK]{second_part}"
    print(text)
    inputs = initial.replace_tokenizer(text, return_tensors='pt')
    mask_index = torch.where(inputs["input_ids"] == initial.replace_tokenizer.mask_token_id)[1].item()

    with torch.no_grad():
        outputs = initial.replace_model(**inputs)
        predictions = outputs.logits[0, mask_index].topk(70)
        predicted_token_ids = predictions.indices.tolist()

    predicted_tokens = initial.replace_tokenizer.convert_ids_to_tokens(predicted_token_ids)
    not_word = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '=', '+', '[', ']', '{', '}', ';', ':', '\'', '\"', ',', '.', '<', '>', '/', '?', '\\', '|', '`', '~']
    pred = []
    ind = 0

    while len(pred) != 5 and ind != len(predicted_tokens):
        ox = 0
        word = predicted_tokens[ind]
        for i in not_word:
            if i in word:
                ox = 1
                break
        # 한글 자음이나 모음만 있는 경우를 제외하고, 한 글자 단어와 word_list에 있는 단어를 제외
        if ox == 0 and len(word) > 1 and word not in word_list and is_hangul_syllable(word):
            pred.append(word)
        ind += 1

    return pred

# 최적의 후보 단어 선택
def select_best_candidate(candidates,sent_1, sent_2):
    input_sentences = [sent_1 + candidate + sent_2 for candidate in candidates]
    tokenized_sentences = [initial.choice_tokenizer(sentence, return_tensors='pt') for sentence in input_sentences]
    outputs = [initial.choice_model(**tokenized_sentence) for tokenized_sentence in tokenized_sentences]
    cls_logits = [output.last_hidden_state[:, 0, :] for output in outputs]
    mean_logits = [torch.mean(logit, dim=-1).item() for logit in cls_logits]
    best_candidate_index = torch.argmax(torch.tensor(mean_logits)).item()
    return candidates[best_candidate_index]

# 음성 학습 및 출력
def voice_return(texts, tts_model, tone_color_converter, speed=1.3):

    # 경로 및 장치 설정
    # ckpt_converter = './checkpoints_v2/converter'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    output_dir = './temp'


    reference_speaker = f'./audio/audio_{n}.wav'  # 복제하려는 음성 파일 경로
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)

    # 텍스트 설정
    # Speed is adjustable
    model = tts_model
    speaker_ids = model.hps.data.spk2id

    for idx, text in enumerate(texts):
        if text in texts[:idx]:
            shutil.copy(f'{output_dir}/audio_token_{texts[:idx].index(text)}.wav', f'{output_dir}/audio_token_{idx}.wav')
        else:
            src_path = output_dir + f"/tmp_{idx}.wav"
            for speaker_key in speaker_ids.keys():
                speaker_id = speaker_ids[speaker_key]
                speaker_key = speaker_key.lower().replace('_', '-')
        
                source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
                model.tts_to_file(text, speaker_id, src_path, speed=speed)
                save_path = f'{output_dir}/audio_token_{idx}.wav'

                # Run the tone color converter
                encode_message = "@MyShell"
                tone_color_converter.convert(
                    audio_src_path=src_path, 
                    src_se=source_se, 
                    tgt_se=target_se, 
                    output_path=save_path,
                    message=encode_message)

    print("음성 변환이 완료되었습니다.")

def speed_up_audio(input_path, output_path, speed_ratio):
    if not (0.5 <= speed_ratio <= 2.0):
        intermediate_path = input_path + "_intermediate.wav"
        if speed_ratio > 2.0:
            first_pass_ratio = 2.0
        else:
            first_pass_ratio = 0.5
        second_pass_ratio = speed_ratio / first_pass_ratio

        command1 = [
            "ffmpeg",
            "-y",  # 덮어쓰기 옵션
            "-i", input_path,
            "-filter:a", f"atempo={first_pass_ratio}",
            "-vn",
            intermediate_path
        ]
        command2 = [
            "ffmpeg",
            "-y",  # 덮어쓰기 옵션
            "-i", intermediate_path,
            "-filter:a", f"atempo={second_pass_ratio}",
            "-vn",
            output_path
        ]
        
        try:
            subprocess.run(command1, check=True, stderr=subprocess.PIPE, text=True)
            subprocess.run(command2, check=True, stderr=subprocess.PIPE, text=True)
            print(f"Audio speed adjusted and saved to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while adjusting audio speed: {e.stderr}")
        finally:
            if os.path.exists(intermediate_path):
                os.remove(intermediate_path)
    else:
        temp_output_path = output_path + "_temp.wav"
        command = [
            "ffmpeg",
            "-y",  # 덮어쓰기 옵션
            "-i", input_path,
            "-filter:a", f"atempo={speed_ratio}",
            "-vn",
            temp_output_path
        ]
        try:
            result = subprocess.run(command, check=True, stderr=subprocess.PIPE, text=True)
            os.replace(temp_output_path, output_path)  # 임시 파일을 최종 파일로 대체
            print(f"Audio speed adjusted and saved to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while adjusting audio speed: {e.stderr}")
        finally:
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)

# 원본 음성과 비슷하도록 가공
def voice_processing(before_audio, new_audio, output_audio):
    initial.separator.separate_to_file(new_audio, initial.temp_path)
    new_path = new_audio[:-4] + "/vocals.wav"

    audio1 = AudioSegment.from_file(before_audio)
    audio2 = AudioSegment.from_file(new_path)
    avg_db2 = audio2.dBFS

    print('[Before_DB] :', Before_DB)
    print('[Avg_db2] :',avg_db2)
    change_in_db = Before_DB - avg_db2
    audio2_adjusted = audio2.apply_gain(change_in_db)
    print('[audio_2_adjusted] :', audio2_adjusted.dBFS)
    
    chunks = split_on_silence(audio2_adjusted, min_silence_len=100, silence_thresh=audio2_adjusted.dBFS)
    processed_sound = AudioSegment.empty()
    for chunk in chunks[:2]:
        processed_sound += chunk

    duration1 = len(audio1)
    duration2 = len(processed_sound)
    rate = duration1 / duration2
    processed_sound.export(output_audio, format="wav")
    if rate < 1:
        speed_up_audio(output_audio, output_audio, 1/rate)

# 기존 음성에 새로운 음성 합성
def replace_audio_segment(input_audio_path, replacement_audio_path, timestamp, output_path):
    original_audio = AudioSegment.from_wav(input_audio_path)
    replacement_audio = AudioSegment.from_wav(replacement_audio_path)

    start_ms = timestamp[0] * 1000
    end_ms = timestamp[1] * 1000
    original_audio = original_audio[:start_ms] + AudioSegment.silent(duration=(end_ms - start_ms)) + original_audio[end_ms:]
    overlay_audio = original_audio.overlay(replacement_audio, position=start_ms)
    original_audio = overlay_audio
    
    original_audio.export(output_path, format="wav")
    
# video 교체
def replace_video(video_path, new_audio_path, output_path):
    try:
        video = VideoFileClip(video_path)
        new_audio = AudioFileClip(new_audio_path)
        video_with_new_audio = video.set_audio(new_audio)
        
        # 비디오를 저장할 때의 파라미터 설정
        video_with_new_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"Video with new audio saved to {output_path}")
    except Exception as e:
        print(f"Error occurred while replacing video audio: {e}")


# temp_path 비우기
def clear_temp_directory():
    temp_path = initial.temp_path
    
    # 디렉토리 내 모든 파일 제거
    for filename in os.listdir(temp_path):
        file_path = initial.temp_path + "/" + filename
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path) 
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
