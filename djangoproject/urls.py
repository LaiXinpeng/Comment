"""djangoproject URL Configuration

The urlpatterns list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,re_path
from app01 import views

urlpatterns = [
    path('admin/', admin.site.urls),  # Django 管理界面路由
    re_path(r'^$', views.user_login),  # 根路径，通常指向用户登录页面
    path('app01/user_login/', views.user_login, name='user_login'),  # 用户登录功能路由
    path('app01/signup_view/', views.signup_view, name='signup_view'),  # 用户注册页面路由
    path('app01/', views.index),  # 应用主页路由
    path('app01/textcategory/', views.text_category),  # 朴素贝叶斯文本分类页面路由
    path('app01/logical/', views.logical),  # 逻辑回归中风风险预测页面路由
    path('app01/mllogical/', views.mllogical),  # 机器学习逻辑回归模型页面路由
    path('app01/mlclassification/', views.mlclassification),  # 机器学习图像分类页面路由
    path('app01/mlpredict/', views.mlpredict),  # 加载图片分类模型（SVM）页面路由
    path('app01/dlclassification/', views.dlclassification),  # 深度学习图像分类页面路由
    path('app01/xlclassification/', views.xlclassification),  # 图像分割页面路由
    path('app01/dlpredict/', views.dlpredict),  # 加载图片分类模型（AlexNet）页面路由
    path('app01/xlpredict/', views.xlpredict),  # 加载图片分割模型Unet页面路由
    path('app01/dldetect/', views.dldetect),  # 深度学习目标检测页面路由
    path('app01/detect/', views.detect),  # 目标检测算法页面路由
    path('app01/pressregression/', views.pressregression),  # 实验室输出电力预测页面路由
    path('app01/pressmlregression/', views.pressmlregression),  # 机器学习电力预测页面路由
    path('app01/video_upload/', views.video_upload),  # 视频上传页面路由
    path('app01/video_detect/', views.video_detect),  # 视频目标检测页面路由
    path('app01/open_face/', views.open_face),  # 人脸识别页面路由
    path('app01/face/', views.face),  # 人脸识别功能路由
]