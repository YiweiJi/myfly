# coding:utf-8
"""
@author:lisheng
Created on 2021/12/15 15:03
"""

import random
import pygame

# 屏幕大小的常量
SCREEN_RECT = pygame.Rect(0, 0, 800, 600)
# 刷新的帧率
FRAME_PER_SEC = 90
# 创建敌机的定时器
CREATE_ENEMY_EVENT = pygame.USEREVENT
# 发射子弹事件
PLANE_FIRE_EVENT = pygame.USEREVENT + 1


class GameSprite(pygame.sprite.Sprite):
    """飞机大战游戏精灵"""

    def __init__(self, image_name, speed=1):
        # 调用父类的初始化方法
        super().__init__()
        # 定义对象的属性
        self.image = pygame.image.load(image_name)
        self.rect = self.image.get_rect()
        self.speed = speed

    def update(self):
        # 在屏幕的垂直方向移动
        self.rect.y += self.speed


class Background(GameSprite):
    """游戏背景精灵"""

    def __init__(self, is_alt=False):
        # 1. 调用父类方法实现精灵的创建
        super().__init__("./images/bg.png")

        # 2. 判断是否交替图像，如果是，需要设置初始位置
        if is_alt:
            self.rect.y = -self.rect.height

    def update(self):
        # 1. 调用父类方法实现
        super().update()

        # 2. 判断是否移除屏幕，如果移除屏幕，将图像设置到屏幕的上方
        if self.rect.y >= SCREEN_RECT.height:
            self.rect.y = -self.rect.height


class Enemy(GameSprite):
    """敌机精灵"""

    def __init__(self):
        # 1. 调用父类方法，创建敌机精灵，同时指定敌机图片
        super().__init__("./images/enemynoob.png")

        # 2. 指定敌机的初始速度 1~3
        self.speed = random.randint(2, 4)

        # 3. 指定敌机的初始位置
        self.rect.bottom = 0

        max_x = SCREEN_RECT.width - self.rect.width
        self.rect.x = random.randint(0, max_x)

    def update(self):
        super().update()

        # 判断是否飞出屏幕，如果是，需要从精灵组删除敌机
        if self.rect.y >= SCREEN_RECT.height:
            self.kill()


class Player(GameSprite):
    """玩家精灵"""

    def __init__(self):
        # 1.调用父类方法，设置玩家图片
        super().__init__("./images/player.png", 0)

        # 2. 设置玩家的初始位置
        self.rect.centerx = SCREEN_RECT.centerx
        self.rect.bottom = SCREEN_RECT.bottom - 5

        # 3. 创建子弹的精灵组
        self.bullets = pygame.sprite.Group()

    def update(self):
        # 玩家在水平方向移动
        self.rect.x += self.speed

        # 控制玩家不离开屏幕
        if self.rect.x < 0:
            self.rect.x = 0
        elif self.rect.right > SCREEN_RECT.right:
            self.rect.right = SCREEN_RECT.right

    def fire(self):
        # 1.创建子弹精灵
        bullet = Bullet()

        # 2. 设置精灵的位置
        bullet.rect.bottom = self.rect.y
        bullet.rect.centerx = self.rect.centerx

        # 3. 将子弹添加到精灵组
        self.bullets.add(bullet)


class Bullet(GameSprite):
    """子弹精灵"""
    def __init__(self):
        # 调用父类方法，设置子弹图片，设置初始速度
        super(Bullet, self).__init__("./images/bullet.png", -2)

    def update(self):
        # 调用父类方法，让子弹沿垂直方向飞行
        super().update()

        # 判断子弹是否飞出屏幕
        if self.rect.bottom < 0:
            self.kill()


class BaseLine(GameSprite):
    """界线"""
    def __init__(self):
        super(BaseLine, self).__init__("./images/line.png")

        # 设置位置
        self.rect.centerx = SCREEN_RECT.centerx
        self.rect.bottom = SCREEN_RECT.bottom - 75
