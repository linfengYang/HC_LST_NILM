# English 
# preparing PLAID and Whited dataset 
PLAID 2018 : https://figshare.com/articles/dataset/PLAID_-_A_Voltage_and_Current_Measurement_Dataset_for_Plug_Load_Appliance_Identification_in_Households/10084619
PLAID 2017: https://figshare.com/articles/dataset/PLAID_2017/11605215
PLAID 2014 : https://figshare.com/articles/dataset/PLAID_2014/11605074
Whitedv1.1 : https://www.cs.cit.tum.de/msrg/resources/

Our study requires TensorFlow 2.5 or higher, as well as TensorFlow Addons.

Then you can generate the VI trajectories (with the appliance name as the foldname), the save path is as follows:
images
  VI_images_test
    2017Sub_one
    2014Sub_one
    2018Sub_one
    2018Agg_one
    Whited

In our experiment, we found that some hyperparameters may greatly affect the identification performance, so, manual adjustment of these hyperparameters (the learning rate, batchsize, etc) may be required in different environments. Recommend to use free GPU resource in Colab environment.

Acknowledgement:
https://github.com/zhuxinqimac/B-CNN
https://github.com/keras-team/keras-io/blob/master/examples/vision/swin_transformers.py

Contact e-mail:
2113301058@st.gxu.edu.cn

# 中文
根据上述链接下载好数据集，然后预处理画出VI轨迹图，以他们电器的名称作为文件夹依次存放.
注意tensorflow版本大于等于2.5.0






