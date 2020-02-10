## 说明

pytorch的pgn
成绩35-40左右分数，不好不差，可以前30吧，不知道前几名80的怎么搞的，应该发现漏洞了？我觉得是
可以多线程预测，速度不慢，1-3小时预测万5w。

---

抛开比赛来看：只能说pgn只能缓解UNK，使用了还是会有不少UNK摘要生成。生成式文本任重道远。


## 数据

数据AutoMaster\**.csv放进data/

predict.sh时指定src/config.py下model_path的路径

## 运行

>1:sh start.sh

>2:sh train.sh

>3:sh predict.sh

## 目录

>data:数据目录

>logs:模型保存目录

>result:输出结果目录

