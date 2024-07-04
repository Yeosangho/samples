## Cifar10 - profiler 테스트 코드 
[원본 코드](https://github.com/kuangliu/pytorch-cifar)

### 파일 설명
- docker-compose.yaml : MPS 및 nsight system 프로파일을 수행할 수 있는 실험 환경 컨테이너를 실행시킵니다. 
- Dockerfile : 실험 환경 컨테이너를 구성하기 위한 이미지를 구성합니다. 이 [도커 파일](https://github.com/ten1010-io/text-generation/blob/main/training/Dockerfile)을 참조하여 구성함.
- main.py : 단일 학습을 위한 엔트리 포인트
- main_dlprof.py : dlprofiling을 위한 학습 실행 엔트리 포인트
- main_dist.py : 분산 학습 수행을 위한 실행 엔트리 포인트 
- ncu.sh : nsight compute profiling을 위한 엔트리 포인트 
- nsys_launch.sh : nsys launch로 프로세스 실행을 위한 쉘 스크립트 
- nsys_interval_record.sh : nsys_launch 이후 주기적 프로파일링을 위한 쉘 스크립트 
- nsys_profile.sh : nsys profile 작업 실행을 위한 쉘 스크립트
- nvml.py : nvml을 통해 프로세스의 gpu 사용량을 확인
