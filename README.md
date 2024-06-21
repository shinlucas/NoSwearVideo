# 유튜브 동영상의 욕설 필터링 및 대체
Replace Swear Voice with natural words and speaker's voice

## 📺 프로젝트 소개
유튜브 동영상의 욕설을 발화자 음성의 자연스러운 단어로 대체하여 불편하지 않게 시청할 수 있도록 제작한 서비스입니다.

## 🔊 상세 소개
- 동영상 시청시 비난하는 어조와 함께 나오는 욕설은 시청자로 하여금 눈쌀을 찌푸리게 하기도 하는데, 이를 대처해보고자 제작했습니다.
- 동영상 내의 욕설을 탐지하고 타임스탬프를 출력합니다.
- 욕설 앞뒤의 문맥을 고려하여 대체어를 생성합니다.
- 대체어에 발화자의 음색 및 어투를 반영하여 대체 음성을 생성합니다.
- 대체 음성을 동영상에 적용합니다.

## 💡 주요 기능
- 메인 페이지 - 로딩 페이지 - 송출 페이지
- 메인 페이지 : 유튜브 링크 입력
- 로딩 페이지 : 욕설필터링 작업을 위한 대기시간 페이지
- 송출 페이지 : 가공된 영상 송출

## 📄 개발환경
- Backend : FLASK (Python)
- 활용 : Whisper, Bert, OpenVoice V2, etc
  
## 📢 주의사항
- 구동시, app.py의 주석을 확인해주세요
- cpu 사용시 다음을 참고(필수, https://github.com/myshell-ai/OpenVoice/pull/262/files)

## 🔍외부링크
- Demo Viceo 1
https://youtu.be/8_fMsp4mNK0
- Demo Viceo 2
https://youtu.be/lXlARboTFlI

