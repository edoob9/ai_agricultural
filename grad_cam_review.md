## < 저자가 논문을 쓰게 된 이유>
CNNs을 기반으로 하는 Deep neural model은 이미지 분류, object detection 등 다양한 CV task에서 훌륭한 성능을 보여줬습니다. 이러한 모델들이 좋은 성능을 가능하게 만들었지만, 이들의 각각의 직관적인 
요소로의 decomposability(모델의 분해가능성)의 부족은 이들을 해석하기 어렵게 만들었습니다.
-> 지능형 시스템에 있어서 신뢰를 구축하기 위해서는 왜 그렇게 예측했는지를 설명할 능력이 있는 <strong>Transparent Models(interpretable model)</strong>이 필요했다. 
아래와 같은 방식으로 설계에 의해 얼마나 이해 및 설명 가능한지 확인할 수 있다.
-  algorithmic transparency은 선형모델이 얼마나 이해하기 쉬우며, 비선형 모델은 조금 더 정교한 모델이 필요
-  decomposabilitysms는 모델 내 각 부분의 설명이 얼마나 intelligent한가다.

### Class Activation Map(CAM)

- CNN + CAM 구조
    - convolution layer와 pooling layer를 활용해서 이미지 내 정보를 요약한다.
    - 마지막(최종) 연산으로 된 feature map이 있는데, GAP를 진행한다.
### <CAM 구조의 한계점>

- GAP layer를 반드시 사용한다
- 뒷부분에 대한 또 다시 fine tuning을 해줘야한다.
- 마지막 convolutional layer에서만 CAM 추출 가능
  Grad-CAM은 여기에서 GAP(Global average pooling)를 사용하지않는다면? 어떻게 할까? 라는 생각에서 진행하는 것이다.

  → feature map별 weight를 학습시킬 수 없다.

  → 그래서, 가중치 구하는 방법을 바꾸자!

  feature map의 각 원소가 특정 class에 주는 영향력을 gradient하는 것이다.=(gradient를 통해서 feature map의 가중치 계산한다.)

CAM과 Grad-CAM의 차이점은 weight를 구하는 방법이 다르다!
  → Grad-CAM은 다른 방법을 사용한다.

how? : GAP 사용 안하고, CNN 구조를 변경하지 않고 사용한다. 그리고 gradient는 특정 class(output)에 특정 input이 주는 영향력이 있다.=(미분의 개념 사용)

따라서 오늘날의 지능형 시스템은 어떠한 경고나 설명 없이 실패하는 경우가 많으며, 이는 사용자가 지능형 시스템의 일관성 없는 output을 보면서 시스템이 왜 그런 의사결정을 했는지에 대해서 궁금하게 됩니다.

Zhou et al. 은 discriminative regions을 식별하고자 fully-connected layer가 없는 image classification CNN에 사용되는 기법인 Class Activation Map(CAP)을 제안

수학적인(수식적으로) 표현한 문장들이 많고, [Learning Deep Features for Discriminative Localization 논문 원본](https://arxiv.org/pdf/1512.04150.pdf)
-> 관련 내용 추후 올리기

- Grad-CAM을 이용해 class discriminative한 localization 방법이다.
- Grad-CAM은 CNN의 마지막 convolutional layer로 흐르는 gradient information을 사용하여 관심이 있는 특정한 의사결정을 위해 각 뉴런에 importance value를 할당하는 방법이다.
- ResNet 등 깊은 모델(neuron을 확인할 수 있는 방법)에 있어서도 인간이 판단할 만한 시각화를 가능케 했고, 특히 이미지 분류 및 VQA등 다양한 downstram task에 대해서 예측을 평가할 수 있는 방법론을 제안했다.

CNN 기반의 모델이 만든 의사결정에 대한 'visual explanations'를 만드는 기술을 제안을 했다 -> feature map을 보여주는 것이며, failure mode에 대한 insight를 제공 
=>논문에서 제시한 Fig. 1에서 visualization output볼 수 있다. 제시한 모델의 grad-cam을 통한 결과를 보면 cat 분류에 대한 설명은 오로지 cat에 해당하는 region만 강조하며 dog 해당하는 region을 강조하지 않습니다!

<strong>??</strong> GuidedBackpropagation에 대한 해석이 필요할 것 같다.

### <확장적인 적용 가능>
Image captioning과 VQA(visual question answering)에 대해서, Grad-CAM은 심지어 non-attention based model도 input image의 discriminative region의 위치를 학습할 수 있음을 보여줍니다.

#### image captioning explained by Grad-CAM

Neuraltalk2 모델에 Grad-CAM을 적용(VGG-16과 LSTM을 적용한 image captioning 모델)

Neuraltalk2가 예측에 대해 정교하게 설명하는 모델이 아님에도 grad-cam을 마지막 CNN-layer에 적용한 class에 대한 로그 확률을 계산하고, caption 예측에 대한 시각적인 해석이 가능하게 했다.
‘man’이라는 caption에 대해 이미지에서 여자가 아닌 남성의 얼굴에만 activated되었음을 확인할 수 있다.

#### VQA explained by Grad-CAM

CNN & RNN을 이용(image representation + question representation) -> 분류 해결했다.
(좀 더 자세히 알아보기)
