# 1. Docker 설치
참고 사이트: https://github.com/sugyeong-jo/Tech/blob/master/Docker/install_for_window.md


# 2. Docker image 가져오기
```bash
docker pull sugyeong/j_python:0.3
docker run -it --volume="C:\Users\USER\OneDrive - UNIST\Attachments:/home/workspace" --name j_python sugyeong/j_python:0.2 /bin/bash
```

# 3. Docker 설치
```bahs
docker login

//Docker 시작
docker start j_python

//Docker 접속
docker attach j_python

//Docker 종료
docker stop j_python
``` 

# 4. Jupyter notebook 실행
```bash
jupyter notebook --allow-root
```

$http://localhost:8888/?tocken=~~$ 복사 후 chrome에 붙여넣기