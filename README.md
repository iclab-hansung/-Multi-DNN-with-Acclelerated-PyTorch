### 모델 추론시 job Queue의 형식(FIFO Queue/Priority Queue) swap 옵션 적용
#### 현재 설정 현황
1. 1종류의 모델 다수를 inference 하면 절반씩나누어 HIGH Stream, low Stream에 할당.
2. Priority Queue는 높은 Priority number가 높은 우선순위를 가진다.
3. FIFO Queue 사용시 모든 Stream은 low Stream, 모든 Priority 값은 0 으로 설정된다.

#### 전체 코드 실행을 위한 실행파일 생성과정
1. CMakeLists.txt가 들어있는 디렉토리에 build 디렉토리 생성
2. build 디렉토리 내에서 cmake.. 명령어 실행
3. 2번 과정 완료 후 make 명령어 실행
4. build 디렉토리 내에 실행파일 생성 완료

#### model_pt 디렉토리를 생성하여 각 모델의 pt파일을 넣어줘야 정상적인 실행 가능
#### - 각 DNN 모델의 version  (model pt파일의 파일 명) 
- alexnet       (alexnet_model.pt)
- densenet201   (densenet_model.pt)
- inception_v3  (inception_model.pt)
- mnasnet1_0    (mnasnet_model.pt)
- mobilenet_v2  (mobile_model.pt)
- resnet152     (resnet_model.pt)
- resnext50     (resnext_model.pt)
- shufflenet_v2_1_0   (shuffle_model.pt)
- squeezenet1_0       (squeeze_model.pt)
- vgg16         (vgg_model.pt)
- wideresnet50  (wideresnet_model.pt)
