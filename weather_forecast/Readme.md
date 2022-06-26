### 기상예측
2021.06 ~ 2022.05 기상자료개방포털에서 제공하는 data를 통해서 이전 데이터를 구축한다. 예측해야할 값은 기상청에서

[기상자료개방포털](https://data.kma.go.kr/cmmn/main.do) 

[기상청](https://www.weather.go.kr/w/index.do) - https://www.weather.go.kr/plus/land/current/aws_table_popup.jsp
AWS 서비스 기반인거 같다.

모은 데이터의 첫날,마지막날의 온도를 통해서 전후 데이터를 알 수 없으니까 제외시킨다.
optimizer은 adam, loss는 후버 손실 MSE와 MAE를 절충한 후버 손실(Huber loss)을 이용!(epochs는 500)
-> 쉽게 말해서 L1 loss과 L2 loss의 장점을 취하면서 단점을 보완하기 위해 제안된 것이 Huber Loss

다음과 같은 결과가 나왔다.
2022-06-25일 날짜를 가지고 오늘 데이터를 예측했다. 근데, 날씨가 오늘 비는 안왔지만, 우중충했다. 분류를 더 세심하게 진행하면 결과가 좋게 나오려나싶다.

<고민>
-> 이전 날 하루를 가지고 예측하지말고, 3일,5일 같이 여러날을 기준으로 예측하면 더 좋은 결과가 나오지않을까?

<strong>야외활동 하기 좋은 날씨입니다. 책상을 벗어나는 하루를 만들어 보는건 어떨까요</strong>
