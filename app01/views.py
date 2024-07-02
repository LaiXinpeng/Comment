import uuid
import joblib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from django.http import HttpResponse
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
import os
import torch
from django.conf import settings
from .unet_model import UNet
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def signup_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        # 检查用户名是否已经存在
        if User.objects.filter(username=username).exists():
            return render(request, 'signup.html', {'error': '用户名已存在，请选择其他用户名'})

        # 创建用户并保存到数据库
        user = User.objects.create_user(username=username, password=password)

        return redirect('user_login')  # 成功注册后重定向到登录页面

    return render(request, 'signup.html')


def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        print(username, password)
        # 在这里对密码进行哈希处理
        # 使用 Django 提供的 authenticate 函数验证用户
        user = authenticate(username=username, password=password)

        if user is not None:
            login(request, user)  # 登录用户
            return render(request, 'index.html')  # 登录成功后的页面，例如首页
        else:
            return render(request, 'login.html', {'error': '用户名或密码错误'})

    return render(request, 'login.html')
def index(request):
    return render(request, 'index.html')
def textcategory(request):
    return render(request, 'text.html')
from app01.predict import classification

def text_category(request):
    print("+++++++++++++++++++++++++++++++++++")
    if request.POST:
        review = request.POST.get('review')  # 使用 .get() 方法获取表单数据
        print("+++++++++++++++++++++++++++++++++++")
        print(review)
        print("------------------------------------")
        predict_res=classification(review)
        emotion=predict_res
        return render(request, 'text.html',{"emotion":emotion})
    return render(request, 'text.html')
def logical(request):
    return render(request, 'logistic_regression.html')

def mllogical(request):
    if request.method == 'POST':
        # 获取表单数据
        sex = request.POST.get('sex')
        age = request.POST.get('age')
        high_press = request.POST.get('high_press')
        heart_disease = request.POST.get('heart_disease')
        marriage = request.POST.get('marriage')
        work_type = request.POST.get('work_type')
        live = request.POST.get('live')
        level = request.POST.get('level')
        bmi = request.POST.get('bmi')
        smoker = request.POST.get('smoker')
        model, scaler = joblib.load('app01/static/Stroke.pkl')
        new_data = {
            'gender': sex,
            'age': age,
            'hypertension': high_press,
            'heart_disease': heart_disease,
            'ever_married': marriage,
            'work_type': work_type,
            'Residence_type': live,
            'avg_glucose_level': level,
            'bmi': bmi,  # 修正为合理的BMI值
            'smoking_status': smoker
        }

        single_data = pd.DataFrame([new_data])
        X_single = single_data[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                                'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]
        print(single_data)

        # 数据标准化（使用训练时相同的Scaler）
        X_single_scaled = scaler.transform(X_single)

        # 使用训练好的模型进行预测
        predicted_probabilities = model.predict_proba(X_single_scaled)  # 预测概率

        # 输出预测概率
        print("Predicted probabilities for the single data sample:")
        print(predicted_probabilities)
        positive_class_probability = predicted_probabilities[0][1]
        print(positive_class_probability)
        # 根据预测的概率，定义中风与否的阈值
        threshold = 0.5
        if positive_class_probability >= threshold:
            print_res = '高'
        else:
            print_res = '低'

        # 返回预测结果到前端页面
        return render(request, 'logistic_regression.html',
                      {'print_res': positive_class_probability, 'stroke_probability': print_res})

    # 如果请求不是POST，则返回空白页面或其他处理
    return HttpResponse("Method Not Allowed")


def mlclassification(request):
    # if request.method == 'POST':
    #     return render(request,'mlcategory.html',{'error_msg':'图片上传成功【OK】'})
    if request.method == 'POST':
        # 获取上传的文件对象
        obj = request.FILES.get('picFile', None)
        if obj:
            # 处理文件信息
            pic_file_name = obj.name
            pic_file_size = obj.size
            pic_file_stuf = os.path.splitext(pic_file_name)[1]

            # 打印文件信息到控制台（可选）
            print('\n上传文件信息：')
            print('-' * 40)
            print('文件名称：{0}'.format(pic_file_name))
            print('文件大小：{0} bytes'.format(pic_file_size))
            print('文件后缀：{0}'.format(pic_file_stuf))


            allowedTypes = ['.png', '.jpg', '.jpeg','.bmp', '.gif']
            if pic_file_stuf not in allowedTypes:
                print('文件类型不正确')
                return render(request, 'mlcategory.html',{'error_msg':'错误：文件类型不正确，请您选择一张正确的图片上传'})
            picUploadUniquename=str(uuid.uuid1())+pic_file_stuf
            print('上传文件唯一名称：{0}'.format(picUploadUniquename))
            # 验证图片的上传路径
            uploadDirPath=os.path.join(os.getcwd(), 'app01/static/images')
            if not os.path.exists(uploadDirPath):
                os.mkdir(uploadDirPath)
                print('服务器上传文件夹创建完毕.')
            else:
                print('服务器上传文件夹已存在.')  # 设置上传文件的全路径

            picFileFullPath=uploadDirPath+os.sep + picUploadUniquename
            print('上传文件全路径：{0}'.format(picFileFullPath))

            print("+++++++++++++++-------------------")
            try:
                with open(picFileFullPath, 'wb+') as fp:

            # 分割上传文件(当上传文件大小2.5MB以上是自动分割)
                    for chunk in obj.chunks():
                        fp.write(chunk)

                    print('[OK]上传文件写入服务器.')  # 设置传递参数

                    context=dict()
                   # 在页面中进行图片的显示
                    img_done_url = 'app01/static/images/' +picUploadUniquename
                    print("+++++++++++++++-------------------")
                    print("+++++++++++++++-------------------")
                    print(img_done_url)
                    print("+++++++++++++++-------------------")
                    print("+++++++++++++++-------------------")
                    context['process_url']= img_done_url[13:]
                    print(context['process_url'])
                    context['success_msg'] = '[OK]图片上传成功！'  # 响应客户端
                    return render(request, 'mlcategory.html', context)
            except:
                print('[Error]上传文件写入服务器失败.')
          # 返回上传成功的消息到前端页面
                return render(request, 'mlcategory.html', {'error_msg': '图片上传成功【OK】'})

        else:
            # 如果没有上传文件，则处理其他逻辑或返回错误消息
            return render(request, 'mlcategory.html', {'error_msg': '请选择要上传的图片'})
    else:
        # 如果不是POST请求，可以处理GET请求的逻辑，例如显示一个空白的上传表单页面
        return render(request, 'mlcategory.html')
def mlpredict(request):
    # 加载svm.model完成图片分类
    if request.POST:
        obj = request.POST.get('path', None)
        image_path = 'app01/' + obj
        import cv2
        img = cv2.imread(image_path) #  读取图片
        print_res = predict_mlcategory_image(img)
        res_category = 'othter'
        if print_res == 0:
            res_category = '黄麻纤维'
        elif print_res == 1:
            res_category = '玉米'
        elif print_res == 2:
            res_category = '大米'
        elif print_res == 3:
            res_category = '甘蔗'
        elif print_res == 4:
            res_category = '小麦'
        context = dict()
        context['print_res'] = res_category
        return render(request,'mlcategory.html',context)

def predict_mlcategory_image(img):
    import cv2
    import numpy as np
    import joblib
    img = cv2.resize(img,(224,224))
    img = np.resize(img,[224*224*1,1])
    img = img.squeeze()
    img = img.reshape(1,-1)
    model = joblib.load('app01/static/svm.pkl')
    img_res = model.predict(img)
    print(img_res)
    return img_res[0]

def dlclassification(request):
    if request.POST:
        # 图片上传的功能，并在页面显示图片
        # 接收客户端请求数据
        obj = request.FILES.get('picFile', None)
        # 处理请求数据
        picFileName = obj.name
        picFileStuff = os.path.splitext(picFileName)[1]
        # 判断上传的文件是否为图片的格式
        allowedTypes = ['.jpg', '.bmp', '.jpeg', '.gif', '.png']
        # 判断上传文件类型是否受限
        if picFileStuff.lower() not in allowedTypes:
            return render(request, 'dlcategory.html',
                          {'error_msg': '错误：文件类型不正确，请您选择一张图片上传!'})
        # 生成唯一的文件名称
        picUploadUniqueName = str(uuid.uuid1()) + picFileStuff
        # 验证图片的上传路径
        uploadDirPath = os.path.join(os.getcwd(), 'app01/static/images')
        if not os.path.exists(uploadDirPath):
            # 创建文件夹
            os.mkdir(uploadDirPath)
            print('服务器上传文件夹创建完毕.')
        else:
            print('服务器上传文件夹已存在.')
        # 设置上传文件的全路径
        picFileFullPath = uploadDirPath + os.sep + picUploadUniqueName
        try:
            # 获取文件的关联并生成文件操作对象fp
            with open(picFileFullPath, 'wb+') as fp:
                # 分割上传文件（当上传文件大小2.5MB以上是自动分割）
                for chunk in obj.chunks():
                    fp.write(chunk)
                print('[OK] 上传文件写入服务器.')
                # 设置传递参数
                context = dict()
                # 在页面中进行图片的显示
                img_done_url = 'app01/static/images/image_done/' + picUploadUniqueName
                context['process_url'] = img_done_url[13:]
                context['success_msg'] = '[OK] 图片上传成功!'
                # 响应客户端
                return render(request, 'dlcategory.html', context)
        except:
            print('[Error] 上传文件写入服务器失败.')
            return render(request, 'dlcategory.html', {'error_msg': '[Error] 图片上传失败!'})
    else:
        return render(request,'dlcategory.html')

#
ResNet_model_path = 'app01/static/ResNet50.pkl'
ResNet_model = joblib.load(ResNet_model_path)

def dlpredict(request):
    if request.method == 'POST':
        obj = request.POST.get('path', None)
        image_path = 'app01/' + obj

        # 读取和处理图像
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))

        # 进行预测
        print_res = predict_dlcategory_image(img)

        # 返回预测结果到模板
        return render(request, 'dlcategory.html',{'print_res': print_res})

    # 如果不是 POST 请求，返回空表单或其他逻辑
    return render(request, 'dlcategory.html')
    # 导入你的UNet模型，确保路径和模型定义正确

def xlpredict(request):
    if request.method == 'POST':
        # 设备选择
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 获取图片路径
        obj = request.POST.get('path', None)
        image_path = 'app01/' + obj
        print(image_path)

        # 加载网络模型
        net = UNet(n_channels=1, n_classes=1)
        net.to(device=device)

        # 加载模型参数
        model_path = os.path.join(settings.MEDIA_ROOT, 'app01/static/best_model.pth')  # 最佳模型的路径，根据实际情况修改
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()

        # 加载图片并进行处理
        img = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.resize(img_gray, (512, 512))
        img_gray = img_gray.reshape(1, 1, img_gray.shape[0], img_gray.shape[1])
        img_tensor = torch.from_numpy(img_gray).to(device=device, dtype=torch.float32)

        # 预测
        with torch.no_grad():
            pred = net(img_tensor)

        # 处理预测结果
        pred = np.array(pred.data.cpu()[0])[0]
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0

        # 保存结果图片
        img_done_url = 'app01/static/images/image_done/' + str(uuid.uuid1()) + '.jpg'
        cv2.imwrite(os.path.join(settings.MEDIA_ROOT, img_done_url), pred)
        # 准备将结果返回到网页
        context = {}
        context['print_res'] = img_done_url[13:]  # 将路径中的前缀部分去掉，以便模板能够正确访问

        # 渲染指定的模板（dldetect.html），并传递上下文数据进行页面显示
        return render(request, 'xlcategory.html', context)
    else:
        # 如果请求不是POST方法，则直接渲染指定的模板（dldetect.html），无需处理数据
        return render(request, 'xlcategory.html')
def xlclassification(request):
    if request.POST:
        # 图片上传的功能，并在页面显示图片
        # 接收客户端请求数据
        obj = request.FILES.get('picFile', None)
        # 处理请求数据
        picFileName = obj.name
        picFileStuff = os.path.splitext(picFileName)[1]
        # 判断上传的文件是否为图片的格式
        allowedTypes = ['.jpg', '.bmp', '.jpeg', '.gif', '.png']
        # 判断上传文件类型是否受限
        if picFileStuff.lower() not in allowedTypes:
            return render(request, 'xlcategory.html',
                          {'error_msg': '错误：文件类型不正确，请您选择一张图片上传!'})
        # 生成唯一的文件名称
        picUploadUniqueName = str(uuid.uuid1()) + picFileStuff
        # 验证图片的上传路径
        uploadDirPath = os.path.join(os.getcwd(), 'app01/static/images')
        if not os.path.exists(uploadDirPath):
            # 创建文件夹
            os.mkdir(uploadDirPath)
            print('服务器上传文件夹创建完毕.')
        else:
            print('服务器上传文件夹已存在.')
        # 设置上传文件的全路径
        picFileFullPath = uploadDirPath + os.sep + picUploadUniqueName
        try:
            # 获取文件的关联并生成文件操作对象fp
            with open(picFileFullPath, 'wb+') as fp:
                # 分割上传文件（当上传文件大小2.5MB以上是自动分割）
                for chunk in obj.chunks():
                    fp.write(chunk)
                print('[OK] 上传文件写入服务器.')
                # 设置传递参数
                context = dict()
                # 在页面中进行图片的显示
                img_done_url = 'app01/static/images/' + picUploadUniqueName
                context['process_url'] = img_done_url[13:]
                context['success_msg'] = '[OK] 图片上传成功!'
                # 响应客户端
                return render(request, 'xlcategory.html', context)
        except Exception as e:
            print('[Error] 上传文件写入服务器失败. ')
            return render(request, 'xlcategory.html', {'error_msg': '[Error] 图片上传失败!'})
    else:
        return render(request,'xlcategory.html')

def dlclassification(request):
    if request.POST:
        # 图片上传的功能，并在页面显示图片
        # 接收客户端请求数据
        obj = request.FILES.get('picFile', None)
        # 处理请求数据
        picFileName = obj.name
        picFileStuff = os.path.splitext(picFileName)[1]
        # 判断上传的文件是否为图片的格式
        allowedTypes = ['.jpg', '.bmp', '.jpeg', '.gif', '.png']
        # 判断上传文件类型是否受限
        if picFileStuff.lower() not in allowedTypes:
            return render(request, 'dlcategory.html',
                          {'error_msg': '错误：文件类型不正确，请您选择一张图片上传!'})
        # 生成唯一的文件名称
        picUploadUniqueName = str(uuid.uuid1()) + picFileStuff
        # 验证图片的上传路径
        uploadDirPath = os.path.join(os.getcwd(), 'app01/static/images')
        if not os.path.exists(uploadDirPath):
            # 创建文件夹
            os.mkdir(uploadDirPath)
            print('服务器上传文件夹创建完毕.')
        else:
            print('服务器上传文件夹已存在.')
        # 设置上传文件的全路径
        picFileFullPath = uploadDirPath + os.sep + picUploadUniqueName
        try:
            # 获取文件的关联并生成文件操作对象fp
            with open(picFileFullPath, 'wb+') as fp:
                # 分割上传文件（当上传文件大小2.5MB以上是自动分割）
                for chunk in obj.chunks():
                    fp.write(chunk)
                print('[OK] 上传文件写入服务器.')
                # 设置传递参数
                context = dict()
                # 在页面中进行图片的显示
                img_done_url = 'app01/static/images/image_done' + picUploadUniqueName
                context['process_url'] = img_done_url[13:]
                context['success_msg'] = '[OK] 图片上传成功!'
                # 响应客户端
                return render(request, 'dlcategory.html', context)
        except:
            print('[Error] 上传文件写入服务器失败.')
            return render(request, 'dlcategory.html', {'error_msg': '[Error] 图片上传失败!'})
    else:
        return render(request,'dlcategory.html')

def predict_dlcategory_image(img):
    img_array = np.expand_dims(img, axis=0)  # 扩展维度，以符合模型输入要求
    img_array = preprocess_input(img_array)

    # 提取特征
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    img_features = base_model.predict(img_array)
    img_features = img_features.reshape(img_features.shape[0], -1)
    img_features = img_features.astype(np.float32) / np.max(img_features)

    # 模型预测
    y_pred = ResNet_model.predict(img_features)[0]  # 返回单个预测结果

    label_names = ['apple', 'avocado', 'banana', 'cherry', 'kiwi', 'mango', 'orange', 'pineapple', 'strawberries',
                   'watermelon']
    # 返回预测的标签名称
    predicted_label = label_names[int(y_pred)]  # 将预测结果转换为整数索引
    print(predicted_label)
    return predicted_label


def dldetect(request):
    if request.POST:
        # 图片上传的功能，并在页面显示图片
        # 接收客户端请求数据
        obj = request.FILES.get('picFile', None)
        # 处理请求数据
        print('kekjmfekif')
        picFileName = obj.name
        picFileStuff = os.path.splitext(picFileName)[1]
        # 判断上传的文件是否为图片的格式
        allowedTypes = ['.jpg', '.bmp', '.jpeg', '.gif', '.png']
        # 判断上传文件类型是否受限
        if picFileStuff.lower() not in allowedTypes:
            return render(request, 'dldetect.html',
                          {'error_msg': '错误：文件类型不正确，请您选择一张图片上传!'})
        # 生成唯一的文件名称
        picUploadUniqueName = str(uuid.uuid1()) + picFileStuff
        # 验证图片的上传路径
        uploadDirPath = os.path.join(os.getcwd(), 'app01/static/images')
        if not os.path.exists(uploadDirPath):
            # 创建文件夹
            os.mkdir(uploadDirPath)
            print('服务器上传文件夹创建完毕.')
        else:
            print('服务器上传文件夹已存在.')
        # 设置上传文件的全路径
        picFileFullPath = uploadDirPath + os.sep + picUploadUniqueName
        try:
            # 获取文件的关联并生成文件操作对象fp
            with open(picFileFullPath, 'wb+') as fp:
                # 分割上传文件（当上传文件大小2.5MB以上是自动分割）
                for chunk in obj.chunks():
                    fp.write(chunk)
                print('[OK] 上传文件写入服务器.')
                # 设置传递参数
                context = dict()
                # 在页面中进行图片的显示
                img_done_url = 'app01/static/images/' + picUploadUniqueName
                context['process_url'] = img_done_url[13:]
                context['success_msg'] = '[OK] 图片上传成功!'
                # 响应客户端
                return render(request, 'dldetect.html', context)
        except:
            print('[Error] 上传文件写入服务器失败.')
            return render(request, 'dldetect.html', {'error_msg': '[Error] 图片上传失败!'})
    else:
        return render(request,'dldetect.html')
def detect(request):
    if request.POST:
        obj = request.POST.get('path', None)
        image_path = 'app01/' + obj

        image = cv2.imread(image_path)
        # image = cv2.resize(image, (224, 224))
        from ultralytics import YOLO
        model = YOLO('app01/static/best.pt')
        # results中包含模型预测之后的边界框和类别
        results = model.predict(image, save=False)  # 加载单张图片完成预测
        # 遍历检测结果
        for result in results:
            for i in range(len(result.boxes)):
                # 获取边界框坐标
                x1, y1, x2, y2 = int(result.boxes.xyxy[i][0]), int(result.boxes.xyxy[i][1]), int(
                    result.boxes.xyxy[i][2]), int(
                    result.boxes.xyxy[i][3])

                # 在图片上绘制边界框和类别标签
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 调整文本位置，确保文本不会超出图片边界
                text_x = max(x1, 10)  # 考虑边界情况
                text_y = max(y1 - 10, 10)  # 考虑边界情况

                # 转换图像格式从OpenCV到Pillow
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                font = ImageFont.truetype('app01/static/simsun.ttc', 80)  # 使用中文字体
                draw = ImageDraw.Draw(image_pil)
                draw.text((text_x, text_y), result.names[int(result.boxes.cls[i])], font=font,
                          fill=(0, 255, 0))  # 文本位置、颜色等属性
                image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        img_done_url = 'app01/static/images/image_done/' + str(uuid.uuid1()) + '.jpg'
        cv2.imwrite(img_done_url, image)  # 将opencv处理后的图片保存在新的路径中
        context = dict()
        context['print_res'] = img_done_url[13:]
        return render(request, 'dldetect.html', context)
    else:
        return render(request, 'dldetect.html')


def pressregression(request):
    return render(request, 'press.html')


from django.shortcuts import render
import joblib
import numpy as np


def pressmlregression(request):
    # 加载预训练模型
    model = joblib.load('app01/static/multivariate_model.pkl')

    if request.method == 'POST':
        # 获取表单数据并进行类型转换
        AT = float(request.POST.get('temperature'))
        V = float(request.POST.get('pressure'))
        AP = int(request.POST.get('humidity'))
        RH = int(request.POST.get('pressure_strength'))

        # 构建输入数据数组
        values = np.array([[AT, V, AP, RH]])

        # 使用模型进行预测
        prediction = model.predict(values)
        prediction = round(prediction[0], 2)

        # 将预测结果传递给模板
        context = {'print_res': prediction}
        return render(request, 'press.html', context)

    # 如果请求不是 POST，可以做一些处理，如返回空页面或错误提示
    return render(request, 'press.html')

def video_upload(request):
    if request.method == 'POST' and request.FILES.get('videoFile'):
        try:
            # 接收上传的视频文件
            video_file = request.FILES['videoFile']

            # 保存视频文件到指定路径（app01/static/videos目录下）
            video_upload_unique_name = str(uuid.uuid4()) + os.path.splitext(video_file.name)[1].lower()
            upload_dir_path = os.path.join(settings.BASE_DIR, 'app01', 'static', 'videos')
            print(upload_dir_path)
            video_file_full_path = os.path.join(upload_dir_path, video_upload_unique_name)

            with open(video_file_full_path, 'wb+') as fp:
                for chunk in video_file.chunks():
                    fp.write(chunk)
            print('[OK] 上传视频文件写入服务器.')

            print(video_upload_unique_name)
            # 设置传递参数
            process_url = settings.STATIC_URL + 'videos/' + video_upload_unique_name

            context = {
                'process_url': process_url,
                'success_msg': '[OK] 视频上传成功!'
            }
            print("-"*40)
            print(process_url)
            print("-"*40)
            # 响应客户端
            return render(request, 'video_upload.html', context)

        except Exception as e:
            print(f'[Error] 上传视频文件写入服务器失败: {e}')
            return render(request, 'video_upload.html', {'error_msg': '[Error] 视频上传失败!'})

    return render(request, 'video_upload.html')
def video_detect(request):
    if request.method == 'POST' and request.POST.get('path'):
        try:
            video_path = request.POST['path']
            video_path = video_path.replace('/static/', 'app01/static/')
            print(video_path)  # 调试用，检查视频路径是否正确
            model = YOLO(os.path.join(settings.BASE_DIR, 'app01', 'static', 'best.pt'))  # 根据实际路径加载模型
            output_dir = os.path.join(settings.BASE_DIR, 'app01', 'static', 'videos_done')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            no_audio = output_dir+'/no_audio.mp4'
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or use "XVID"
            out = cv2.VideoWriter(no_audio, fourcc, fps, (width, height))
            while cap.isOpened():
                success, frame = cap.read()
                if success:
                    results = model(frame)
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                else:
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()

            clip = VideoFileClip(video_path)
            output_clip = VideoFileClip(no_audio)
            output_clip = output_clip.set_audio(clip.audio)
            output_clip.write_videofile(output_dir+'/out_audio.mp4', audio_codec='aac')

            context = {
                'success_msg': '[OK] 视频处理成功!',
                'process_url2': '/static/videos_done'+'/out_audio.mp4'
            }
            return render(request, 'video_upload.html', context)

        except Exception as e:
            print(f'[Error] 处理视频文件失败: {e}')
            return render(request, 'video_upload.html', {'error_msg': '[Error] 视频处理失败!'})

    return render(request, 'video_upload.html')

def open_face(request):
    if request.method == 'POST':
        # Load a model
        model = YOLO('app01/static/face_best.pt')  # load an official detection model
        results = model.predict(source="0", show=True, tracker="app01/static/fruits.yaml")

    return render(request, 'face.html')

def face(request):
        return render(request, 'face.html')

