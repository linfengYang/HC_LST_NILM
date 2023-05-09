# English 
# preparing PLAID and Whited dataset 

PLAID 2018 : https://figshare.com/articles/dataset/PLAID_-_A_Voltage_and_Current_Measurement_Dataset_for_Plug_Load_Appliance_Identification_in_Households/10084619

PLAID 2017: https://figshare.com/articles/dataset/PLAID_2017/11605215

PLAID 2014 : https://figshare.com/articles/dataset/PLAID_2014/11605074

Whitedv1.1 : https://www.cs.cit.tum.de/msrg/resources/

Our study requires tensorFlow 2.5 or higher, as well as tensorFlow-addons.
Our enivornment (for reference only)：
tensorflow 2.6.0; tensorFlow-addons 0.18.0; keras 2.6.0; numpy 1.23.1

Then you can generate the VI trajectories (with the appliance name as the foldname), the save path is as follows:

 images
  
   VI_images_test
    
     2017Sub_one
    
     2014Sub_one
    
     2018Sub_one
    
     2018Agg_one
     
     Whited

In our experiment, we found that some hyperparameters may greatly affect the identification performance, so, manual adjustment of these hyperparameters (the learning rate, batchsize, etc) may be required in different environments. Recommend to use free GPU resource in Colab environment.

 '''
  A problem has occured when I try to save the best model to conduct the predict, 
  hence the final results are lower than theory (this can be seen by observing the last layer accuracy fluctuating in training). 
  Our paper use the mean value of 3 experiment results as the final results.
 '''

Contact e-mail:
2113301058@st.gxu.edu.cn

# 中文
根据上述链接下载好数据集，然后预处理画出VI轨迹图，以他们电器的名称作为文件夹依次存放(预处理好的数据集可能稍后会公开).
注意tensorflow版本大于等于2.5.0
由于环境可能不同，因此，可能需要手动调整一些超参数(如学习率变化，batchsize等)以产生更好的识别结果。
对于batchsize，实验发现16，32，64，128产生的结果会有些许变化，因此如果GPU显存大于10G的话推荐先采用128，根据结果看是否需要进一步减小。
推荐使用Colab免费平台及Autodl平台

由于主要使用的是10折交叉验证进行模型评估，因此每一折训练完后，照理应该保存最优模型再进行这一折的测试，但在保存模型时出现了一个暂未解决的bug，
因此最终产生的结果肯定是比理论值更低的，原本想拿几个一区的论文进行比较，由于此bug并未解决，因此只能采用本实验的结果，欢迎解决的大佬在评论区留言或私信

验证对VI轨迹做的预处理在不同的评估方法上（K折，8：2）得到的结果会有差异，主要表现在：K折缩小了差距（因为9折训练，1折测试）

这个VI轨迹其实有个问题，就是样本数太少，我也看过一些研究采用一个CSV画多个VI轨迹图，但根据我实验发现，这会使得整体指标都往上提高不少（也很容易解释，一个CSV画出来好几个长得很容易区分的VI轨迹，虽然增大了数据量，但“不公平”），因此个人不推荐使用此方法。
所以对于VI轨迹图数量较少，模型会更容易过拟合，因此本文注释处原本采取了三个数据增强方法（Flip,Crop,Rotation）,但训练的epoch会增大几倍（大概需要180轮），得到的结果确是相近的。因此本文最终只采用RandomFlip，根据论文中画出的acc曲线可以看出，存在过拟合的趋势，但是性价比很高。

总而言之，论文尚有诸多小问题尚待优化，欢迎批评指正。
Contact e-mail:
2113301058@st.gxu.edu.cn
