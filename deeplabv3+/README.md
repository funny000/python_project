模型上传压缩包f分为两部分：
1.modelUpload.json 
	说明上传模型预测代码的相关内容，包括必填项和选填项。
	必填项：
		name:  模型名称，例如：武汉天河机场民航客机目标识别
		weight：模型训练的结果权重参数文件,例如：airplane.th 。
	选填项：
		icon:记录图片名称,根据模型类别的不同，图片名称不同，名称不可变换
			语义分割：
				image.png 原始图片名称
				label.png 标签图片名称
				pred.png  预测图片名称
			目标识别：
				pred.png  预测图片名称
			变化检测：
				A.png	  变化前图片名称
				B.png	  变化后图片名称
				label.png 标签图片名称
				pred.png  预测图片名称
		evaluate：评价指标，按照json中样式进行填写
			description：说明对应指标中文名称
			best：说明改模型对应的评价指标数值
			
			
	详细信息可查看modelUpload.json文件
	相关书写规范需严格按照json格式
	
2.model文件夹
	存放模型预测代码及权重文件
	
	注：1. 权重文件名称需与modelUpload.json中完全一致,并存放在"weight"文件夹中；
		2. 预测模型图片存放在"icon"文件夹中，若无相关图片展示，则删除改文件夹；
		3. 预测文件Predict.py、PredictModel.py，文件名称不可更改；
		4. Predict.py 模型评估。输入图片，进行预测后生成图片，保存并可查看；
			输入参数：image_path(图片路径)、weight_path(权重文件路径)、gpu_num(使用gpu个数)
			输出：语义分割、变化检测：ndarray数组
				  目标识别：geojson 包括 类别,概率,矩形坐标
		5. PredictModel.py  模型预测。输入图片路径，输出的是预测后的数组或目标识别预测后的geojson文件
			输入参数：image_path(图片路径)、weight_path(权重文件路径)、gpu_num(使用gpu个数)
			输出：语义分割、变化检测、目标识别：生成图片
		6. 预测代码结构如实例所示
		