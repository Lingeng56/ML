#%%
#%%
import random
random_ = random.randint(0, 100)
while True:
    guess = int(input("please input your number:"))
    if guess > random_:
        print("your number is bigger")
    elif guess < random_:
        print("your number is smaller")
    elif guess == random_:
        print("you are right")
        break
# 第一次循环之前 我们先生成随机数
#              再输入我们猜的数字
# 开始第一次循环 判断guess 是否等于 random_ 基本就不相等 符合了while后面的命题
# 开始分支判断了 如果guess 大于 random_ 就打印 your number is bigger
# 如果guess 小于 random_ ....
# 第二次循环