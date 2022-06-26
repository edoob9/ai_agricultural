import tensorflow as tf
import numpy as np

# 기상청에서 얻은 데이터이다.
TABLE_DATA = """시:분	강수	강수15	강수60	강수3H	강수6H	강수12H	일강수	기온	풍향1	풍속1(m/s)	풍향10	풍속10(m/s)
23:00	○	0	0	0	0	0	0	25.2	210.9	SSW	3.4	207.8	SSW	3.1	86
22:00	○	0	0	0	0	0	0	25.5	206.9	SSW	3.4	197.0	SSW	3.6	83
21:00	○	0	0	0	0	0	0	25.6	223.6	SW	2.6	189.9	S	2.8	83
20:00	○	0	0	0	0	0	0	26.4	168.4	SSE	2.1	188.6	S	3.3	77
19:00	○	0	0	0	0	0	0	27.5	187.0	S	2.3	200.7	SSW	3.2	71
18:00	○	0	0	0	0	0	0	28.8	193.7	SSW	4.3	203.8	SSW	2.8	71
17:00	○	0	0	0	0	0	0	29.9	261.8	W	2.9	230.7	SW	2.2	71
16:00	○	0	0	0	0	0	0	29.3	238.7	WSW	3.9	228.6	SW	2.8	73
15:00	○	0	0	0	0	0	0	29.9	209.5	SSW	3.6	212.5	SSW	3.8	71
14:00	○	0	0	0	0	0	0	30.7	242.9	WSW	2.5	240.9	WSW	2.9	68
13:00	○	0	0	0	0	0	0	30.4	212.8	SSW	2.1	206.0	SSW	3.4	69
12:00	○	0	0	0	0	0	0	30.3	199.5	SSW	2.8	194.1	SSW	2.3	65
11:00	○	0	0	0	0	0	0	27.8	145.0	SE	3.0	143.3	SE	1.8	70
10:00	○	0	0	0	0	0	0	25.1	172.5	S	1.5	159.4	SSE	1.7	83
09:00	○	0	0	0	0	0	0	24.1	139.1	SE	1.4	147.4	SSE	1.7	84
08:00	○	0	0	0	0	0	0	23.3	128.1	SE	1.1	150.8	SSE	1.5	86
07:00	○	0	0	0	0	0	0	22.5	182.5	S	1.9	169.9	S	1.9	91
06:00	○	0	0	0	0	0	0	21.6	186.0	S	1.0	174.5	S	1.3	98
05:00	○	0	0	0	0	0	0	21.3	205.2	SSW	2.5	203.9	SSW	2.1	98
04:00	○	0	0	0	0	0	0	21.1	212.7	SSW	2.7	203.3	SSW	2.2	98
03:00	○	0	0	0	0	0	0	20.8	152.1	SSE	1.1	166.2	SSE	2.0	99
02:00	○	0	0	0	0	0	0	21.2	203.1	SSW	1.5	197.3	SSW	1.8	99
01:00	○	0	0	0	0	0	0	21.1	208.7	SSW	2.2	208.5	SSW	2.8	99
00:00	○	0	0	0	0	0	17.5	21.3	197.0	SSW	2.6	205.7	SSW	3.2	99"""
MODEL_PATH = './data/model.h5'


def extract_temperature_from_table():
    data = TABLE_DATA.split('\n')
    data = data[1:]
    data = [float(x.split('\t')[8]) for x in data[::-1]]
    return np.asarray(data)


def forecast(temperature):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    predict = model.predict(temperature[np.newaxis])[0][0]
    if predict > 30:
        print("날씨가 많이 더울 예정입니다. 야외활동에 주의하세요")
    elif predict > 25:
        print("날씨가 더울 예정입니다. 옷을 가볍게 입고 나가세요")
    elif predict > 15:
        print("야외활동 하기 좋은 날씨입니다. 책상을 벗어나는 하루를 만들어 보는건 어떨까요")
    elif predict > 5:
        print("선선한 날씨입니다. 가벼운 외투 하나더 챙겨 입는것을 추천드립니다")
    elif predict > -5:
        print("날씨가 추울 예정입니다. 옷을 두껍게 입고 나가세요")
    elif predict > -15:
        print("날씨가 많이 추울 예정입니다. 야외활동에 주의하세요")


if __name__ == '__main__':
    temperature = extract_temperature_from_table()
    forecast(temperature)

