"""
creaTED
for using flyai lab
except that  are useless
"""
from flyai.train_helper import *
# 在项目下创建submit.py文件，然后运行即可提交。
# 系统设定不会提交data文件夹
submit(train_name="my_train",code_path=os.curdir, cmd='python main.py')

# #或者提交zip代码压缩包
# submit(train_name="my_train", code_path="D:/code/my_train_code.zip", cmd='python main.py')