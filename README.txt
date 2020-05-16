		中兴算法大赛设计文档
1.为解决数据不平衡问题(训练集正例为负例的1.3倍),尝试过采用交叉熵损失函数的class_weight项，或者在预测时调阈值(阈值取预测概率的中位数)。
2.分别预训练word2vec,glove,fasttext词向量，并以此训练ESIM模型
3.训练ESIM模型时，使用十折验证，最后将十折预测结果求均值，作为该词向量以及相应词向量训练出来的模型的预测结果。
4.最后将单模型训练出来的A榜分数超过86.8的预测结果进行投票，最终A榜得分88.867.(单模型最高分87.783，word2vec效果最好）
5.用于投票的模型有以下九个：
    1.word2vec，不删除低频词，预训练20轮，lr:0.0004,dropout:0.5,batch_size:32,hiddensize:300   87.400
    2.word2vec，删除频词低于5的低频词，预训练50轮，lr:0.0002,dropout:0.3,batch_size:32,hiddensize:300  87.583
    3.word2vec，删除低于5的低频词，预训练100轮，lr:0.0002,dropout:0.3,batch_size:32,hiddensize:300  87.783（使用了class_weight）
    4.glove，删除低于5的低频词，预训练50轮，lr:0.0002,dropout:0.5,batch_size:32,hiddensize:300  
    5.fasttext，不删除低频词，预训练20轮，lr:0.0004,dropout:0.5,batch_size:32,hiddensize:300  
    6.word2vec，删除频词低于5的低频词，预训练50轮，lr:0.0002,dropout:0.3,batch_size:32,hiddensize:300  使用增强后的文本训练
    7.word2vec，删除频词低于5的低频词，预训练50轮，lr:0.0002,dropout:0.3,batch_size:32,hiddensize:400  使用增强后的文本训练
 （剩下的记不清是哪个了，我在提交结果里面挑出来九个进行投票的，最高分是87.783，上面七个投票最低有88.5分）
   （只保存了单模型最高的权重，glove的训练使用的是GitHub上面的代码，我存放在data里面）
6.使用文本数据增强。A=B,B=C -->A=C ,A=B,B≠C -->A≠C (最终生成了20多万数据，总共取了40万数据进行训练，效果并不好。
7.训练集中有13000多条文本对是一模一样的，尝试过将其剔除，以此减轻数据不平衡的问题，效果并非很好，但参与了最后的投票集成。
 （具体的数据分析，可以看数据预处理以及数据分析.ipynb)
8.使用了自集成，加入第10个epoch是验证精度最高的模型，我取8，9，10，11，12的预测结果投票，分数从87.583提升至87.650.
9.最终取了已提交的几个文件投票，存放在vote文件夹里面，投票后分数达到88.867。按上述1-7步生成的预测文件，投票结果一般在88.583之上。

代码使用说明：
1.所需的包见requirements.txt
2.如果想要重新训练词向量，可以在终端运行：python run_pretraining.py 
   参数默认为训练20轮，50轮，100轮的word2vec，20轮的fasttext。
   在 data/wordvector/中可以看到生成的词向量。
3.数据预处理是生成词典，并且结合上一步生成的词向量生成embedding矩阵。
   可在终端运行  python data_process.py --word_vector_dir=data/wordvector/w2v.txt ^
				--embedding_matrix_dir=data/embedding_matrix_w2v.npy ^
				--word2indices_dir=data/word_to_indices_w2v.file
    在data/中可以看到生成的文件。
     word_vector_dir为词向量存放的地址，embedding_matrix_dir嵌入矩阵存放的地址，word2indices_dir为词表存放的地址
4.可以在train_config.py中配置训练的参数，然后直接在终端输入 python run_train.py即可。
5.可以在终端直接运行 python run_predict.py --predict_dir=predict.txt  ^
				      --weight_dir=weights/best.pth.tar  ^
                                                                      --word2indices_dir=data/word_to_indices_w2v100.file
   predict_dir为预测文件存放的地址，weight_dir为模型权重存放的的地址，word2indices_dir词表存放的地址
6.将预测好的文件放在一个文件夹里面，并对其进行投票集成，可以运行如下命令：
                       python  vote.py  --input_dir=vote  --outputdir=predict.txt
     其中input_dir是存储预测结果的的文件夹，output_dir是输出投票后的结果的地址




