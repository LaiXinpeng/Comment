import jieba
import joblib

label_map = {
    'news_story': '民生故事',
    'news_culture': '文化',
    'news_entertainment': '娱乐',
    'news_sports': '体育',
    'news_finance': '财经',
    'news_house': '房产',
    'news_car': '汽车',
    'news_edu': '教育',
    'news_tech': '科技',
    'news_military': '军事',
    'news_travel': '旅游',
    'news_world': '国际',
    'stock': '股票',
    'news_agriculture': '三农',
    'news_game': '游戏',
}

# 将英文标签转换成中文函数
def translate_label(label):
    return label_map.get(label, label)  # 如果找不到对应的中文标签，则返回原始英文标签

def classification(text):
    words = jieba.cut(text)
    TF = joblib.load('app01/static/count_vec.pkl')
    model = joblib.load('app01/static/bayes.pkl')
    s = "".join(words)
    test_features = TF.transform([s])
    predict_label = model.predict(test_features)[0]
    predict_label = translate_label(predict_label)
    return predict_label
