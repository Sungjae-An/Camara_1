# 문자열
""" 문자열 """
''' 문자열 '''

# 샵 뒤에 쓰면 주석.
# 파일 맨위, 함수 바로 아래 or Class 바로 아래파일 맨 위에 있으면 설명(docstring)으로 인식된다.
# 설명(docstring)은 VScode에서 마우스 올리면 설명이 뜨게하는것.
# 설명(docstring)은 #주석과 달리 여러줄 쓰기가 편리한 장점.

### Miniconda (Anaconda Prompt)

conda create -n exercise1 python=3.11
conda activate exercise1
pip install pyrealsense2 opencv-python mediapipe numpy
conda list

conda remove -n exercise1 --all  # 환경 지울때.
conda info --envs # 환경 어떤것들 만들었나 확인용.


### VScode + GitHub

# 프로젝트 폴더 불러오기
# 프로젝트 폴더내 data, notebooks, src 폴더 생성.
# src폴더내 __init__.py 파일생성.

__pycache__/
*.pyc
.ipynb_checkpoints/
data/raw/
.DS_Store
Thumbs.db
# 프로젝트 폴더에 .gitignore 생성하고 내부에 위의 내용 쳐라.

# GitHub 에서 repository 생성 후 (프로젝트별로 생성) VScode에서 연결
git remote add origin https://github.com/Sungjae-An/Camera_1.git

# GitHub 연결 안될때
git remote set-url origin https://github.com/Sungjae-An/Camera_1.git

# Git clone으로 프로젝트폴더 복사
git clone https://github.com/Sungjae-An/CAMERA_1.git

# Git 다양한 명령어들
git init #최초 프로젝트 생성시에만. 이 폴더를 git이 관리한다는 뜻.

git --version
git remote -v # 현재 쓰는 origin 저장소들이 뭐가있나 보여줌
git status

# ★★★★ 환경을 VScode에서 만드는 과정이 꼭 필요. (환경 만들기)
conda env export --no-builds > camara_1.yml # 프로젝트 폴더에 camera_1.yml 생성

# ★★★★Python interpreter 선택해라! (환경 고르기) 밑의 방법으로 확인해라.
where python # 경로에 miniconda3\envs\exercise1 포함되면 성공.

git pull # ★★★★ 중요 !!!

# ★★★★ Git 작업 끝날때 꼭 commit + push
git add. 
git commit -m "Initial project structure"

git branch -M main
git push -u origin main # git push하면 main으로 push됨.

# Import - basic
from src.camera import Camera
from src.utils import load_data

# Import - use __init__.py
from src import Camera, load_data

##  다른 컴퓨터에서 할 것 순서.
1. Miniconda, VS code, Git 설치
2. "Anaconda prompt" 열고 
    cd /d D:\BB_coding\Camara_1
    conda conda env create -f camara_1.yml
    conda activate exercise1
3. git clone https://github.com/Sungjae-An/CAMARA_1.git
4. cd Camara_1 # 이미 프로젝트 폴더내면 안해도됨.
5. git config --global user.name "Sungjae An"
git config --global user.email "annuguri88@gmail.com" #commit전 최초한번

# git config 잘 되었나 확인: 
git config --global user.name
git config --global user.email

ddd